import asyncio
import contextlib
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import re
from textwrap import dedent
import io
import json
import time
from collections import defaultdict
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, model_validator
import sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import numpy as np
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    ModelSettings,
    function_tool,
)
from agents.memory.sqlite_session import SQLiteSession
from agents.run import ModelInputData, CallModelData
import shap
import statsmodels as sm
import scipy as spy
import xgboost as xgb
import optuna

from rogueone.utils import llm_cfg, rouge_one_cfg, test_agent_cfg
from rogueone.utils.console import (
    ConsoleManager,
    tool_feedback,
    truncate_tool_output,
)
from rogueone.utils.prompt_dump import dump_prompt
from rogueone.llm.agents.core.knowledge import backend as knowledge_backend
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.wandb_utils import (
    log_fold_payload,
    safe_wandb_log,
    wandb_logging_wrapper,
)
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.prompts import set_phased as _set_phased
from rogueone.utils import topology as _topology
from rogueone.llm.agents.web_search import WebSearchAgent
from rogueone.llm.agents.core.tester.metrics import (
    FeatureMetricProfile,
    compute_feature_metrics,
    _N_JOBS as _METRICS_N_JOBS,
)
from rogueone.llm.agents.core.tool_error_reporting import ToolErrorReporter, report_bypass
from rogueone.utils import instrumentation
from rogueone.utils.fold_safe_features import seeded_attribute_mask
from rogueone.llm.clients import create_openai_client_for_vllm
from rogueone.utils.sandbox import sandbox_scope
from rogueone.utils import scientist_signal as _sci_signal
from rogueone.utils import fx_switches
from rogueone.utils.pandas_exec import needs_unescape

FOLD_RETRY_LIMIT = 1
# Per-call thread cap for XGBoost CPU training. Mirrors the cap used for
# permutation_importance in metrics.py so outer parallelism × inner BLAS
# threads stays under the OpenBLAS 128-region limit.
_XGB_N_JOBS = _METRICS_N_JOBS


def _normalize_user_command(command: str) -> str:
    """Normalize a user-supplied exec() snippet so accidental formatting does not break it.

    Mirrors the helper in :mod:`rogueone.llm.agents.core.extractor.agent` so the Tester
    and Scientist agents tolerate the same multi-statement / partially-indented payloads.
    Handles literal ``\\n`` / ``\\t`` escape sequences, semicolon-then-indent, plain
    indented-continuation, and uniform block indentation. See the Extractor module for
    the full rationale and the runs that surfaced each pattern.
    """
    if not isinstance(command, str):
        return command
    if ("\\n" in command or "\\t" in command) and needs_unescape(command):
        command = command.replace("\\n", "\n").replace("\\t", "\t")
    normalized = dedent(command).strip()
    normalized = re.sub(r";[ \t]*\n[ \t]+", "\n", normalized)
    if "\n" in normalized:
        try:
            compile(normalized, "<tester-command>", "exec")
        except IndentationError:
            no_blocks = not any(
                ln.rstrip().endswith(":") for ln in normalized.splitlines()
            )
            if no_blocks:
                normalized = "\n".join(
                    ln.lstrip() for ln in normalized.splitlines()
                )
        except SyntaxError:
            pass
    return normalized


def build_python_executor_scope(df_attributes, assessment_model) -> dict:
    """Exec scope of ``generic_python_executor_tool`` (single dict: globals == locals).

    ``ROGUE_ONE_SANDBOX=on`` restricts builtins and hides module file I/O (E13);
    ``off`` returns exactly the pre-E13 dict. Labels reach this scope only as the
    in-memory ``df_attributes['target']`` column (read once from ``labels.csv`` by
    ``TrainTestSplitter`` at start-up); the scope gets no path or file handle.
    """
    scope = {
        "df_attributes": df_attributes,
        "assessment_model": assessment_model,
        "pd": pd,
        "np": np,
        "sklearn": sklearn,
        "shap": shap,
        "sm": sm,
        "spy": spy,
        "xgb": xgb,
    }
    return sandbox_scope(scope)


def _build_tester_model_settings(cfg) -> "ModelSettings":
    """Build a :class:`ModelSettings` from the Tester's :class:`AgentConfig`.

    Includes ``max_tokens`` only when configured.
    """
    kwargs = {
        "temperature": cfg.temperature,
        "reasoning": cfg.reasoning,
    }
    if getattr(cfg, "max_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return ModelSettings(**kwargs)


def _enforce_pruning_floor(
    df_attribute_explanations: "pd.DataFrame",
    status_before: "pd.Series",
    newly_pruned: list[str],
    shap_lookup: dict[str, float],
    mi_lookup: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Restore the highest-signal newly-pruned features when active count is below the floor.

    The floor is ``max(3, ceil(0.2 * pre_active))`` where ``pre_active`` is the number of
    Active features just before the LLM-driven prune was applied. When restoration is
    needed, features are ranked by ``(mean_abs_shap, mutual_information)`` descending and
    restored until the floor is met.

    Mutates ``df_attribute_explanations`` in place. Returns ``(remaining_newly_pruned, restored)``.
    """
    # E6: seeded raw-column rows (augment) are unprunable and do not count toward the floor.
    agent_rows = ~seeded_attribute_mask(df_attribute_explanations)
    pre_active = int(((status_before != "Pruned") & agent_rows).sum())
    if pre_active == 0:
        return newly_pruned, []
    floor = max(3, (pre_active + 4) // 5)
    active_after = int(
        ((df_attribute_explanations["Status"] != "Pruned") & agent_rows).sum()
    )
    if not newly_pruned or active_after >= floor:
        return newly_pruned, []

    ranked_to_keep = sorted(
        newly_pruned,
        key=lambda name: (
            shap_lookup.get(name, 0.0),
            mi_lookup.get(name, 0.0),
        ),
        reverse=True,
    )
    n_to_unprune = floor - active_after
    restore = ranked_to_keep[:n_to_unprune]
    if not restore:
        return newly_pruned, []

    mask = df_attribute_explanations["Attribute"].isin(restore)
    df_attribute_explanations.loc[mask, "Status"] = "Active"
    if "Pruned Iteration" in df_attribute_explanations.columns:
        df_attribute_explanations.loc[mask, "Pruned Iteration"] = pd.NA
    if "Prune Rationale" in df_attribute_explanations.columns:
        df_attribute_explanations.loc[mask, "Prune Rationale"] = pd.NA
    remaining = [n for n in newly_pruned if n not in set(restore)]
    return remaining, restore


def _safe_mean(values: list[float]) -> float:
    """Calculate the mean of a list, returning NaN if list is empty.

    Args:
        values (list[float]): List of numeric values.

    Returns:
        float: Mean of values, or NaN if list is empty.
    """
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float:
    """Calculate the standard deviation of a list, returning NaN if list is empty.

    Args:
        values (list[float]): List of numeric values.

    Returns:
        float: Standard deviation of values, or NaN if list is empty.
    """
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


#: Run-total counter of fold matrices the inf guard had to clean.
INF_GUARD_COUNTER = "inf_guard"


def _inf_guard(x: Any) -> np.ndarray:
    """Clean a fold matrix the way the scoring fits already do (A17 relaunch).

    A feature replayed on one partition can be infinite there although it was finite on
    the full frame, and a float64 value beyond float32 range becomes infinite inside
    XGBoost. The scoring fits pass every matrix through ``np.nan_to_num(float32,
    posinf=1e6, neginf=-1e6)``, but the assessment fit and the pruning tools received
    the raw matrix, so one such value raised ``XGBoostError: Input data contains inf``
    and ended the run without a result. This applies the same clipping (NaN is left
    alone: XGBoost treats it as missing). Only under ``ROGUE_ONE_INF_GUARD``.
    """
    arr = np.asarray(x)
    try:
        arr = arr.astype(np.float32)
    except (TypeError, ValueError):
        return np.asarray(x)
    if np.isinf(arr).any():
        instrumentation.count(INF_GUARD_COUNTER)
        arr = np.nan_to_num(arr, nan=np.nan, posinf=1e6, neginf=-1e6)
    return arr


def _inner_cv_score(
    X_train: Any,
    y_train: Any,
    base_params: dict[str, Any],
    task_type: str,
    n_splits: int = 4,
    seed: int = 42,
    balance_weights: bool = False,
) -> float | None:
    """Score this iteration's feature set by K-fold CV *inside the training portion*.

    This is the selection signal for nested cross-validation: the reported test fold
    never participates, so choosing an iteration by this value cannot contaminate the
    reported metric. The model trained here is discarded -- the caller still reports the
    metric from a model fitted on the full training portion.

    For regression the raw RMSE is returned rather than a normalised value; only the
    ordering across iterations matters for selection, and a shared denominator would not
    change it.

    Args:
        X_train (Any): Feature matrix for the training portion (4 outer folds).
        y_train (Any): Targets for the training portion. For classification these must
            already be label-encoded integers.
        base_params (dict[str, Any]): XGBoost params, identical to the reported model's.
        task_type (str): Either "classification" or "regression".
        n_splits (int): Number of inner folds.
        seed (int): Seed for the inner split, kept deterministic across iterations.
        balance_weights (bool): Apply balanced sample weights, mirroring the outer fit.

    Returns:
        float | None: Mean inner-fold score, or None when the split is not viable.
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train).ravel()
    if X.shape[0] < n_splits * 2 or y.shape[0] != X.shape[0]:
        return None

    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    is_classification = task_type == "classification"

    try:
        if is_classification:
            # Stratify when every class can populate each inner fold; otherwise a
            # stratified split raises and a plain KFold is the safe fallback.
            _, class_counts = np.unique(y, return_counts=True)
            splitter = (
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if class_counts.min() >= n_splits
                else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            )
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    except Exception as split_exc:
        print(f"[inner_cv] split not viable, no selection signal: {split_exc!r}")
        return None

    scores: list[float] = []
    failures: list[str] = []
    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(splits):
        # One unusable inner fold must not discard the whole selection signal, so each
        # fold is scored independently and the mean is taken over those that succeed.
        try:
            if is_classification:
                # XGBoost requires labels to be exactly 0..k-1 for the classes it is
                # fitted on. An unstratified inner split (used when some class is
                # rarer than n_splits) can leave a class entirely out of the training
                # half, which used to raise "Invalid classes inferred from unique
                # values of `y`" and silently void the signal for the whole dataset.
                # Re-index to the classes actually present, then map predictions back;
                # validation rows whose class is absent from training simply count as
                # errors, which is the honest outcome.
                present = np.unique(y[inner_train_idx])
                local = np.searchsorted(present, y[inner_train_idx])
                model = xgb.XGBClassifier(**base_params)
                weights = (
                    class_weight.compute_sample_weight("balanced", y=local)
                    if balance_weights
                    else None
                )
                model.fit(X[inner_train_idx], local, sample_weight=weights)
                predicted = present[
                    np.asarray(model.predict(X[inner_val_idx])).astype(int)
                ]
                scores.append(float(accuracy_score(y[inner_val_idx], predicted)))
            else:
                model = xgb.XGBRegressor(**base_params)
                model.fit(X[inner_train_idx], y[inner_train_idx])
                scores.append(
                    float(
                        np.sqrt(
                            mean_squared_error(
                                y[inner_val_idx], model.predict(X[inner_val_idx])
                            )
                        )
                    )
                )
        except Exception as fold_exc:
            failures.append(f"fold {fold_idx}: {fold_exc!r}")

    if failures:
        print(
            f"[inner_cv] {len(failures)}/{len(splits)} inner folds failed, scoring on "
            f"the remaining {len(scores)}: {'; '.join(failures[:2])}"
        )
    if not scores:
        # The caller falls back to the legacy selection signal when this is None.
        print("[inner_cv] every inner fold failed, no selection signal available")
        instrumentation.note_inner_cv(None)
        return None
    instrumentation.note_inner_cv(scores)
    value = float(np.mean(scores))
    return value if np.isfinite(value) else None


@dataclass
class TestResultRegression:
    """Container for regression model evaluation results across K-fold validation.

    Stores computed regression metrics (MSE, MAE, R2), fold-level performance data,
    and execution status/failure information.

    Attributes:
        mses (list[float]): Mean squared error for each fold.
        maes (list[float]): Mean absolute error for each fold.
        r2s (list[float]): R-squared scores for each fold.
        max_val (float): Maximum value in the target variable.
        min_val (float): Minimum value in the target variable.
        mean_val (float): Mean value of the target variable.
        report (str): Text summary of the evaluation.
        status (str): Execution status ("completed", "aborted", "skipped").
        dataset_id (str): Identifier for the dataset being evaluated.
        fold_seed (int): Seed used for deterministic fold generation.
        fold_metrics (list[dict], optional): Per-fold performance metrics and metadata.
        aborted_fold_id (int, optional): Index of fold where execution was aborted.
        skip_reason (str, optional): Reason dataset was skipped without evaluation.
    """

    mses: list[float]
    maes: list[float]
    r2s: list[float]
    max_val: float
    min_val: float
    mean_val: float
    report: str = ""
    status: str = "completed"
    dataset_id: str = ""
    fold_seed: int = 42
    fold_metrics: list[dict[str, Any]] | None = None
    aborted_fold_id: int | None = None
    skip_reason: str | None = None
    inner_cv_RMSE: float | None = None
    train_R2: float | None = None

    @property
    def value_range(self) -> float:
        return abs(self.max_val - self.min_val)

    @property
    def n_rmse(self) -> float:
        return _safe_mean(self.n_rmses)

    @property
    def n_rmse_std(self) -> float:
        return _safe_std(self.n_rmses)

    @property
    def n_rmses(self) -> list[float]:
        return [rmse / self.mean_val for rmse in self.rmses]

    @property
    def rmses(self) -> list[float]:
        return [np.sqrt(mse) for mse in self.mses]

    @property
    def rmse(self) -> float:
        return _safe_mean(self.rmses)

    @property
    def rmses_std(self) -> float:
        return _safe_std(self.rmses)

    @property
    def mse(self) -> float:
        return _safe_mean(self.mses)

    @property
    def mse_std(self) -> float:
        return _safe_std(self.mses)

    @property
    def mae(self) -> float:
        return _safe_mean(self.maes)

    @property
    def mae_std(self) -> float:
        return _safe_std(self.maes)

    @property
    def r2(self) -> float:
        return _safe_mean(self.r2s)

    @property
    def r2_std(self) -> float:
        return _safe_std(self.r2s)

    @property
    def as_dict(self) -> dict:
        if self.status != "completed":
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
                "report": self.report,
            }
        return {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "fold_seed": self.fold_seed,
            "aborted_fold_id": self.aborted_fold_id,
            "skip_reason": self.skip_reason,
            "n_RMSE": self.n_rmse,
            "RMSE": self.rmse,
            "MSE": self.mse,
            "MAE": self.mae,
            "R2": self.r2,
            "inner_cv_RMSE": self.inner_cv_RMSE,
            "train_R2": self.train_R2,
            "report": self.report,
        }

    def to_dict(self, only_metrics: bool = True) -> dict:
        if self.status != "completed":
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
            }
        if only_metrics:
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
                "n_RMSE": self.n_rmse,
                "MSE": self.mse,
                "MAE": self.mae,
                "R2": self.r2,
            }
        else:
            return self.as_dict


@dataclass
class TestResultClassification:
    """Container for classification model evaluation results across K-fold validation.

    Stores computed classification metrics (accuracy, precision, recall, F1, AUROC),
    fold-level performance data, and execution status/failure information.

    Attributes:
        accuracys (list[float]): Accuracy scores for each fold.
        precisions (list[float]): Precision scores for each fold.
        recalls (list[float]): Recall scores for each fold.
        f1s (list[float]): F1 scores for each fold.
        aurocs (list[float]): Area under ROC curve for each fold.
        report (str): Text summary of the evaluation.
        status (str): Execution status ("completed", "aborted", "skipped").
        dataset_id (str): Identifier for the dataset being evaluated.
        fold_seed (int): Seed used for deterministic fold generation.
        fold_metrics (list[dict], optional): Per-fold performance metrics and metadata.
        aborted_fold_id (int, optional): Index of fold where execution was aborted.
        skip_reason (str, optional): Reason dataset was skipped without evaluation.
    """

    accuracys: list[float]
    precisions: list[float]
    recalls: list[float]
    f1s: list[float]
    aurocs: list[float]
    report: str = ""
    status: str = "completed"
    dataset_id: str = ""
    fold_seed: int = 42
    fold_metrics: list[dict[str, Any]] | None = None
    aborted_fold_id: int | None = None
    skip_reason: str | None = None
    inner_cv_accuracy: float | None = None
    train_accuracy: float | None = None

    # feature_importance: dict[str, float]
    @property
    def accuracy(self) -> float:
        return _safe_mean(self.accuracys)

    @property
    def accuracy_std(self) -> float:
        return _safe_std(self.accuracys)

    @property
    def precision(self) -> float:
        return _safe_mean(self.precisions)

    @property
    def precision_std(self) -> float:
        return _safe_std(self.precisions)

    @property
    def recall(self) -> float:
        return _safe_mean(self.recalls)

    @property
    def recall_std(self) -> float:
        return _safe_std(self.recalls)

    @property
    def f1(self) -> float:
        return _safe_mean(self.f1s)

    @property
    def f1_std(self) -> float:
        return _safe_std(self.f1s)

    @property
    def auroc(self) -> float:
        return _safe_mean(self.aurocs)

    @property
    def auroc_std(self) -> float:
        return _safe_std(self.aurocs)

    @property
    def as_dict(self) -> dict:
        if self.status != "completed":
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
                "report": self.report,
            }
        return {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "fold_seed": self.fold_seed,
            "aborted_fold_id": self.aborted_fold_id,
            "skip_reason": self.skip_reason,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "AUROC": self.auroc,
            "inner_cv_accuracy": self.inner_cv_accuracy,
            "train_accuracy": self.train_accuracy,
            "report": self.report,
        }

    @property
    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.as_dict])

    def to_dict(self, only_metrics: bool = True) -> dict:
        if self.status != "completed":
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
            }
        if only_metrics:
            return {
                "status": self.status,
                "dataset_id": self.dataset_id,
                "fold_seed": self.fold_seed,
                "aborted_fold_id": self.aborted_fold_id,
                "skip_reason": self.skip_reason,
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "AUROC": self.auroc,
            }
        else:
            return self.as_dict

    def __repr__(self):
        return dedent(f"""
            Accuracy: {self.accuracy}
            Precision: {self.precision}
            Recall: {self.recall}
            F1 Score: {self.f1}
            AUROC: {self.auroc}
            Report: \n{self.report}
        """)


class CodeInput(BaseModel):
    """Schema for Python code execution requests with optional reasoning.

    Allows passing code as a plain string (defaults reasoning to empty) or as a
    structured dict with code and reasoning fields.

    Attributes:
        code (str): The Python code to execute.
        reasoning (str): Explanation of why this code execution is needed.
    """

    code: str = Field(..., description="The python code to execute.")
    reasoning: str = Field(
        "",
        description="The reasoning behind the code execution. Explain why this code is being executed and what it aims to achieve.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, v: object) -> object:
        """Allow the model to pass code_input as a plain string instead of a CodeInput dict."""
        if isinstance(v, str):
            # A tool parser that does not follow "$ref" hands the object over as JSON text.
            if rouge_one_cfg.inline_tool_schema_refs and v.lstrip().startswith("{"):
                try:
                    parsed = json.loads(v)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
                    return parsed
            return {"code": v, "reasoning": ""}
        return v


class TesterAgent:
    """Agent responsible for evaluating feature quality through comprehensive testing.

    Tests extracted features for predictive power, feature importance, statistical relationships,
    and robustness. Uses K-fold cross-validation across regression or classification tasks.
    Can optionally prune features that don't contribute meaningfully to predictions.

    Attributes:
        cfg_experiment (ExperimentConfig): Experiment configuration including task type and data paths.
        include_report (bool): Whether to generate detailed test reports.
        fold_count (int): Number of K-fold splits to use.
        fold_seed (int): Random seed for deterministic fold generation.
        fold_retry_limit (int): Maximum retry attempts for failed fold trials.
        knowledge_agent (WebSearchAgent): Agent for literature searches during testing.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        include_report: bool = True,
        enable_wandb: bool | None = None,
    ):
        """Initialize the TesterAgent.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.
            include_report (bool, optional): Whether to generate test reports. Defaults to True.
        """
        self.cfg_experiment = cfg
        self.include_report = include_report
        self.fold_count = rouge_one_cfg.k_folds
        self.fold_seed = rouge_one_cfg.fold_seed
        self.fold_retry_limit = min(
            max(rouge_one_cfg.fold_retry_limit, 0), FOLD_RETRY_LIMIT
        )
        self.enable_wandb = (
            rouge_one_cfg.do_logging if enable_wandb is None else enable_wandb
        )
        self.knowledge_agent = WebSearchAgent(cfg=self.cfg_experiment)
        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._tool_error_reporter: ToolErrorReporter | None = None
        self._assessment_report_produced = False

    @staticmethod
    def prune_attributes_in_df(
        df: pd.DataFrame, df_attribute_explanations: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove pruned attributes from the feature dataframe.

        Filters out features marked as "Pruned" in the attribute explanations.

        Args:
            df (pd.DataFrame): Feature dataframe to prune.
            df_attribute_explanations (pd.DataFrame): DataFrame with attribute metadata including Status.

        Returns:
            pd.DataFrame: Dataframe with pruned attributes removed.
        """
        pruned_attributes = df_attribute_explanations[
            df_attribute_explanations["Status"] == "Pruned"
        ]["Attribute"].tolist()

        df_pruned = df.drop(columns=pruned_attributes, errors="ignore")

        return df_pruned

    @staticmethod
    def _rebuild_after_pruning(
        df_attributes: pd.DataFrame,
        X_test_all_features: np.ndarray | None,
        feature_names: list[str],
        X_train_prev: np.ndarray,
        y_train_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Rebuild (X_train, y_train, X_test) from the post-pruning active set.

        ``df_attributes`` is the training frame after the pruning stage (active
        feature columns plus ``target``); ``X_test_all_features`` is the full test
        matrix with one column per entry of ``feature_names``. If pruning would leave
        no active feature the previous matrices are kept.
        """
        active = [c for c in df_attributes.columns if c != "target"]
        if not active:
            return X_train_prev, y_train_prev, X_test_all_features
        X_train = df_attributes[active].to_numpy()
        y_train = df_attributes["target"].to_numpy()
        X_test = X_test_all_features
        if X_test is not None:
            X_test = X_test[:, [feature_names.index(name) for name in active]]
        return X_train, y_train, X_test

    @staticmethod
    def _metric_profiles_to_markdown(
        metric_profiles: list[FeatureMetricProfile],
    ) -> str:
        header = (
            "| feature_name | mutual_information | vif | pfi_mean | pfi_std | "
            "noise_sensitivity | mean_abs_shap | warnings |\n"
            "|---|---:|---:|---:|---:|---:|---:|---|"
        )

        def _fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.6g}"

        rows = []
        for profile in metric_profiles:
            warnings = (
                "; ".join(profile.computation_warnings)
                if profile.computation_warnings
                else ""
            )

            rows.append(
                "| {name} | {mi} | {vif} | {pfi_m} | {pfi_s} | {noise} | {shap_val} | {warn} |".format(
                    name=profile.feature_name,
                    mi=_fmt(profile.mutual_information),
                    vif=_fmt(profile.vif),
                    pfi_m=_fmt(profile.permutation_importance_mean),
                    pfi_s=_fmt(profile.permutation_importance_std),
                    noise=_fmt(profile.noise_sensitivity),
                    shap_val=_fmt(profile.mean_abs_shap),
                    warn=warnings,
                )
            )
        return "\n".join([header, *rows]) if rows else header

    @staticmethod
    def _metric_profiles_to_dataframe(
        metric_profiles: list[FeatureMetricProfile],
    ) -> pd.DataFrame:
        def _fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.6g}"

        rows: list[dict[str, str]] = []
        for profile in metric_profiles:
            warnings = (
                "; ".join(profile.computation_warnings)
                if profile.computation_warnings
                else ""
            )
            rows.append(
                {
                    "feature_name": profile.feature_name,
                    "mutual_information": _fmt(profile.mutual_information),
                    "vif": _fmt(profile.vif),
                    "pfi_mean": _fmt(profile.permutation_importance_mean),
                    "pfi_std": _fmt(profile.permutation_importance_std),
                    "noise_sensitivity": _fmt(profile.noise_sensitivity),
                    "mean_abs_shap": _fmt(profile.mean_abs_shap),
                    "warnings": warnings,
                }
            )

        columns = [
            "feature_name",
            "mutual_information",
            "vif",
            "pfi_mean",
            "pfi_std",
            "noise_sensitivity",
            "mean_abs_shap",
            "warnings",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _assessment_report_path(self, step: int | None) -> Path:
        filename = f"feature_assessment_step_{step if step is not None else 0}.md"
        output_dir = Path(self.cfg_experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    @staticmethod
    def _write_assessment_report(path: Path, markdown: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    async def test_hypotheses(
        self,
        df_attributes_folds: list[pd.DataFrame],
        df_attribute_explanations: pd.DataFrame,
        *,
        step: int | None = None,
        trial_index: int | None = None,
        shared_session=None,
    ) -> tuple[TestResultClassification | TestResultRegression, pd.DataFrame]:
        """Execute hypothesis testing on extracted features using K-fold cross-validation.

        Tests features for predictive power on the target variable. For classification or
        regression tasks with tabular data. Generates metrics and optional
        detailed assessment report.

        Args:
            df_attributes_folds (list[pd.DataFrame]): List of dataframes, one per fold.
            df_attribute_explanations (pd.DataFrame): DataFrame containing feature explanations and metadata.
            step (int, optional): Fold-run identifier used for test-fold selection.
            trial_index (int, optional): Global trial index (0..N-1) used as the WandB step
                so that per-trial metrics are plotted consecutively.
            shared_session (topology.SharedIterationSession, optional): E9f
                ``set_phased`` only. Stage A and Stage B run in the iteration's shared
                session with the neutral system prompt and the ASSESS / PRUNE phase
                messages. ``None`` (default) is Full-v2.

        Returns:
            tuple: (TestResult object with metrics, DataFrame with updated attribute explanations)
        """
        _t0 = time.perf_counter()

        def _log(msg: str) -> None:
            elapsed = time.perf_counter() - _t0
            ConsoleManager.console_print(
                f"[cyan][TesterAgent step={step} trial={trial_index} "
                f"+{elapsed:6.2f}s][/cyan] {msg}"
            )

        _log(
            f"test_hypotheses: start "
            f"(folds={len(df_attributes_folds)}, "
            f"include_report={self.include_report}, "
            f"task_type={self.cfg_experiment.task_type})"
        )

        x_folds: list[np.ndarray] = []
        y_folds: list[np.ndarray] = []
        X_test: np.ndarray | None = None
        y_test: np.ndarray | None = None
        selected_test_fold_id = 0
        # E14: outer-train / outer-test row ids (matrix order) for the L1 split check.
        l1_train_ids: list | None = None
        l1_test_ids: list | None = None

        match self.cfg_experiment.modality:
            case "tabular":
                if len(df_attributes_folds) != self.fold_count:
                    raise ValueError(
                        f"Expected exactly {self.fold_count} folds, got {len(df_attributes_folds)}"
                    )

                fold_sizes = [len(fold) for fold in df_attributes_folds]
                if any(size == 0 for size in fold_sizes):
                    raise ValueError(f"One or more folds are empty: {fold_sizes}")

                if "id" in df_attributes_folds[0].columns:
                    fold_ids = [
                        set(fold["id"].tolist()) for fold in df_attributes_folds
                    ]
                    all_test_ids = set().union(*fold_ids)
                    total_rows = sum(len(ids) for ids in fold_ids)
                    if total_rows != len(all_test_ids):
                        raise ValueError(
                            "Fold coverage invalid: duplicate ids detected across test folds."
                        )

                if (
                    rouge_one_cfg.scientist_signal == _sci_signal.INNER_CV_L1
                    and "id" in df_attributes_folds[0].columns
                ):
                    _l1_test_fold = (step or 0) % len(df_attributes_folds)
                    l1_test_ids = df_attributes_folds[_l1_test_fold]["id"].tolist()
                    l1_train_ids = [
                        _id
                        for _fid, _fold in enumerate(df_attributes_folds)
                        if _fid != _l1_test_fold
                        for _id in _fold["id"].tolist()
                    ]

                df_attributes_folds = [
                    d.drop("id", axis=1) for d in df_attributes_folds
                ]
                feature_names = [
                    c for c in df_attributes_folds[0].columns if c != "target"
                ]

                x_folds = [d[feature_names].to_numpy() for d in df_attributes_folds]
                if rouge_one_cfg.inf_guard:
                    x_folds = [_inf_guard(x) for x in x_folds]
                y_folds = [d["target"].to_numpy() for d in df_attributes_folds]

                selected_test_fold_id = (step or 0) % len(x_folds)
                train_fold_ids = [
                    fold_id
                    for fold_id in range(len(x_folds))
                    if fold_id != selected_test_fold_id
                ]
                if not train_fold_ids:
                    raise ValueError("Unable to form a train set from tabular folds.")

                X_train_assessment = np.vstack(
                    [x_folds[fold_id] for fold_id in train_fold_ids]
                )
                y_train_assessment = np.hstack(
                    [y_folds[fold_id] for fold_id in train_fold_ids]
                )
                X_test = x_folds[selected_test_fold_id]
                y_test = y_folds[selected_test_fold_id]

                df_attributes = pd.DataFrame(X_train_assessment, columns=feature_names)
                df_attributes["target"] = y_train_assessment

            case _:
                raise NotImplementedError(
                    f"Modality {self.cfg_experiment.modality} not supported yet."
                )

        if X_train_assessment.shape[0] == 0:
            raise ValueError(
                "Training-only subset is empty; cannot run assessment/pruning."
            )

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )
        active_feature_names = [c for c in df_attributes.columns if c != "target"]

        if len(active_feature_names) == 0:
            raise ValueError("No active features available for assessment.")

        X_train_assessment = df_attributes[active_feature_names].to_numpy()
        y_train_assessment = df_attributes["target"].to_numpy()
        X_train = X_train_assessment
        y_train = y_train_assessment

        # When some features are marked Pruned, X_train uses only active_feature_names
        # but x_folds (and therefore X_test) still contains ALL feature columns.
        # Filter X_test to the same active columns to prevent XGBoost shape mismatches
        # on model.predict() that silently produce NaN accuracy for every post-prune iteration.
        X_test_all_features = X_test
        if X_test is not None and len(active_feature_names) < len(feature_names):
            active_col_indices = [
                feature_names.index(name) for name in active_feature_names
            ]
            X_test = X_test[:, active_col_indices]

        _log(
            f"prepared training matrices: X_train={X_train_assessment.shape}, "
            f"active_features={len(active_feature_names)}, "
            f"X_test={None if X_test is None else X_test.shape}"
        )

        if self.cfg_experiment.task_type == "classification":
            label_encoder = LabelEncoder()
            y_assessment_model = (
                np.asarray(label_encoder.fit_transform(y_train_assessment))
                .ravel()
                .astype(int)
            )
            unique_classes = np.unique(y_assessment_model)
            clf_params: dict[str, Any] = {
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": self.fold_seed,
                "device": "cpu",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "n_jobs": _XGB_N_JOBS,
            }
            if len(unique_classes) > 2:
                clf_params.update(
                    {
                        "objective": "multi:softprob",
                        "num_class": int(len(unique_classes)),
                    }
                )
            else:
                clf_params.update({"objective": "binary:logistic"})

            assessment_model: xgb.XGBClassifier | xgb.XGBRegressor = xgb.XGBClassifier(
                **clf_params
            )
            assessment_model.fit(X_train_assessment, y_assessment_model)
        else:
            y_assessment_model = y_train_assessment
            assessment_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                random_state=self.fold_seed,
                device="cpu",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=_XGB_N_JOBS,
            )
            assessment_model.fit(X_train_assessment, y_assessment_model)

        _log("assessment model fit complete; computing pre-computed feature metrics...")

        metric_profiles = compute_feature_metrics(
            X_train=X_train_assessment,
            y_train=y_assessment_model,
            feature_names=active_feature_names,
            model=assessment_model,
            task_type=self.cfg_experiment.task_type,
            fold_seed=self.fold_seed,
        )

        _log(f"pre-computed metrics ready ({len(metric_profiles)} feature profiles).")

        metric_warning_records: list[dict[str, str]] = []
        for profile in metric_profiles:
            if profile.computation_warnings:
                warning_text = "; ".join(profile.computation_warnings)
                metric_warning_records.append(
                    {
                        "feature": profile.feature_name,
                        "warnings": warning_text,
                    }
                )
                ConsoleManager.console_error_print(
                    f"Metric warning for {profile.feature_name}: {warning_text}"
                )

        metric_table_markdown = self._metric_profiles_to_markdown(metric_profiles)
        metric_table_df = self._metric_profiles_to_dataframe(metric_profiles)
        ConsoleManager.print_dataframe_as_table(
            metric_table_df,
            title="Pre-computed attribute metrics",
            style="cyan",
        )
        assessment_report_path = self._assessment_report_path(step)

        # Ensure pruning metadata columns exist for downstream status tracking.
        if "Pruned Iteration" not in df_attribute_explanations.columns:
            df_attribute_explanations["Pruned Iteration"] = pd.NA
        if "Prune Rationale" not in df_attribute_explanations.columns:
            df_attribute_explanations["Prune Rationale"] = pd.NA

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._num_features_pruned = 0
        self._pruning_decisions: list[dict[str, str]] = []
        self._assessment_stage_status = "completed"
        self._pruning_stage_status = "completed"
        self._assessment_report_produced = False
        self._tool_error_reporter = ToolErrorReporter(
            agent_name=self.__class__.__name__,
            output_dir=self.cfg_experiment.output_dir,
            session_name=f"tester_agent_step_{step or 0}",
        )

        _note_book = ""

        def _report_tool_error(
            *,
            tool_name: str,
            message: str,
            exception: Exception,
            code: str | None = None,
        ) -> str:
            self._tool_call_counter_failed[tool_name] += 1
            if self._tool_error_reporter is not None:
                self._tool_error_reporter.report(
                    tool_name=tool_name,
                    message=message,
                    exception=exception,
                    step=step,
                    context={"code": code} if code is not None else None,
                )
            return tool_feedback(ConsoleManager.console_error_print(message, code=code))

        @function_tool
        async def search_in_literature_tool(question: str) -> str:
            """
            Ask the Knowledge Agent to look up questions related to feature testing in from the literature.
            This includes general questions about what methodology exists for performing certain types of tests.
            And also specific questions about how certain types of features have been evaluated in the literature.
            And how to compare different methodologies for evaluating features.

            Args:
                question (str): The question to look up.

            Returns:
                str: An answer to the question rooted in the literature.
            """

            try:
                self._tool_call_counter["search_in_literature_tool"] += 1
                res = await self.knowledge_agent.explain_query(question, caller_role="tester")

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "methodology_lookup_tool",
                    f"Question: {question}\nAnswer: {res[:500]}...",
                )

                self._tool_call_counter_success["search_in_literature_tool"] += 1
                return res
            except Exception as e:
                return _report_tool_error(
                    tool_name="search_in_literature_tool",
                    message=f"Error looking up methodology: {e}",
                    exception=e,
                )

        @function_tool
        async def take_note_tool(note: str) -> str:
            """Use this tool to note down important discoveries and facts that should be included in the final rapport.
            The notes can be retrieved later using the 'get_notes_tool'.

            Args:
                note (str): Note to take.

            Returns:
                str: Confirmation message.
            """
            try:
                self._tool_call_counter["take_note_tool"] += 1
                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "take_note_tool",
                    f"Taking note: {note}",
                )

                nonlocal _note_book
                _note_book += f"{note}\n"

                self._tool_call_counter_success["take_note_tool"] += 1
                return "Note added successfully."
            except Exception as e:
                return _report_tool_error(
                    tool_name="take_note_tool",
                    message=f"Error taking note: {e}",
                    exception=e,
                )

        @function_tool
        async def get_notes_tool() -> str:
            """Use this tool to retrieve all notes taken so far using the 'take_note_tool'.


            Returns:
                str: All notes taken so far.

            """

            try:
                self._tool_call_counter["get_notes_tool"] += 1
                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "get_notes_tool",
                    "Getting all notes",
                )

                self._tool_call_counter_success["get_notes_tool"] += 1
                return _note_book
            except Exception as e:
                return _report_tool_error(
                    tool_name="get_notes_tool",
                    message=f"Error getting notes: {e}",
                    exception=e,
                )

        @function_tool
        async def get_precomputed_metrics_tool() -> str:
            """Return pre-computed feature metric profiles as a markdown table."""
            try:
                self._tool_call_counter["get_precomputed_metrics_tool"] += 1
                self._tool_call_counter_success["get_precomputed_metrics_tool"] += 1
                return metric_table_markdown
            except Exception as e:
                return _report_tool_error(
                    tool_name="get_precomputed_metrics_tool",
                    message=f"Error loading pre-computed metrics: {e}",
                    exception=e,
                )

        @function_tool
        async def generic_python_executor_tool(code_input: CodeInput) -> str:
            """This tool is a general python execution tool. It can be used to set up experiments to assess the usefulness of attributes.
            The environment has access to a attribute data (df_attributes) as pandas DataFrames.
            The environment also has access to the trained assessment model (assessment_model), an XGBoost model
            already fitted on the training data. You can use this model to generate predictions, extract feature
            importances, compute SHAP values, or perform any other model-based analysis to better understand
            the derived features.
            All code is non persistent and stateless between calls.

            The environment have the following libraries imported and available for use:
            - pandas as pd
            - numpy as np
            - sklearn
            - shap
            - statsmodels as sm
            - scipy as spy
            - xgboost as xgb

            The following variables are available in the execution environment:
            - df_attributes: pandas DataFrame with feature columns and a 'target' column.
            - assessment_model: a fitted XGBoost model (XGBClassifier or XGBRegressor) trained on the current feature set.

            The dataframe has the following structure:
            - Each row represents an instance.
            - Each column represents a feature/attribute, except for the 'target' column which is the target variable to predict.

            The output of the code should be assigned to a variable named 'output'.

            Returns:
                str: The output of the executed code or an error message if execution fails.
            """

            try:
                self._tool_call_counter["generic_python_executor_tool"] += 1

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    # Single namespace dict used as both globals and locals so
                    # nested lambdas / defs in multi-statement code can see
                    # variables defined at the top level of the snippet
                    # (otherwise lambda __globals__ != exec locals dict and
                    # we get spurious NameErrors). See _exec_user_command in
                    # the Extractor module for the same fix.
                    scope = build_python_executor_scope(
                        df_attributes, assessment_model
                    )
                    exec(
                        _normalize_user_command(code_input.code),
                        scope,
                    )

                    output = scope.get("output", "No output variable defined.")

                output = f"{output}"

                max_output_length = min(len(output), 500)

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "generic_python_executor_tool",
                    message=code_input.reasoning,
                    code=code_input.code,
                    post_message=(
                        f"Output: {output[:max_output_length]}" + "..."
                        if len(output) > max_output_length
                        else ""
                    ),
                )

                self._tool_call_counter_success["generic_python_executor_tool"] += 1
                return truncate_tool_output(output)

            except Exception as e:
                return _report_tool_error(
                    tool_name="generic_python_executor_tool",
                    message=f"Error executing the code. Please check your syntax and try again: {e}",
                    exception=e,
                    code=code_input.code,
                )

        @function_tool
        async def attribute_lookup_tool(attribute_names_list: list[str]) -> str:
            """
            Look up the explanation for a given list of attribute names from the attribute_explanations dictionary.

            Args:
                attribute_names_list (list[str]): The names of the attributes to look up.

            Returns:
                str: The explanation for the attribute, or a message indicating it was not found.
            """

            try:
                self._tool_call_counter["attribute_lookup_tool"] += 1
                res = df_attribute_explanations[
                    df_attribute_explanations["Attribute"].isin(attribute_names_list)
                ]

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "attribute_lookup_tool",
                    f"Lookup attributes: {attribute_names_list}",
                )

                self._tool_call_counter_success["attribute_lookup_tool"] += 1
                return res
            except Exception as e:
                return _report_tool_error(
                    tool_name="attribute_lookup_tool",
                    message=f"Error looking up attributes: {e}",
                    exception=e,
                )

        @function_tool
        async def attribute_pruning_tool(attribute_names_list: list[str]) -> str:
            """
            Prune the given list of attribute names from the attribute_explanations dictionary.

            Args:
                attribute_names_list (list[str]): The names of the attributes to prune.

            Returns:
                str: A message indicating the result of the pruning operation.
            """

            try:
                self._tool_call_counter["attribute_pruning_tool"] += 1

                if not self._assessment_report_produced:
                    self._tool_call_counter_failed["attribute_pruning_tool"] += 1
                    report_bypass(self._tool_error_reporter, site="v3", tool_name="attribute_pruning_tool", message="Pruning is blocked: the feature assessment report has not been produced yet.", step=step, kind="pruning_blocked", agent_name="TesterAgent")
                    return (
                        "ERROR: Pruning is blocked. The feature assessment report has not been "
                        "produced yet. You must complete the assessment stage before any "
                        "pruning decisions can be made."
                    )

                # E6: seeded raw-column rows (augment) are protected from pruning.
                _seeded = set(
                    df_attribute_explanations.loc[
                        seeded_attribute_mask(df_attribute_explanations), "Attribute"
                    ].astype(str)
                )
                _protected = [n for n in attribute_names_list if str(n) in _seeded]
                attribute_names_list = [
                    n for n in attribute_names_list if str(n) not in _seeded
                ]

                self._num_features_pruned += (
                    df_attribute_explanations.loc[
                        df_attribute_explanations["Attribute"].isin(
                            attribute_names_list
                        ),
                        "Status",
                    ]
                    .ne("Pruned")
                    .sum()
                )

                df_attribute_explanations.loc[
                    df_attribute_explanations["Attribute"].isin(attribute_names_list),
                    "Status",
                ] = "Pruned"

                df_attribute_explanations.loc[
                    df_attribute_explanations["Attribute"].isin(attribute_names_list),
                    "Pruned Iteration",
                ] = f"Trial {trial_index if trial_index is not None else (step if step is not None else 0)}"

                # remove rows where Attribute is in attribute_names_list

                ConsoleManager.console_agent_logging(
                    "TesterAgent",
                    "attribute_pruning_tool",
                    f"Pruned attributes ({len(attribute_names_list)}): {attribute_names_list}",
                )

                self._tool_call_counter_success["attribute_pruning_tool"] += 1
                if _protected:
                    return (
                        f"Attributes pruned successfully: {attribute_names_list}. "
                        f"Not pruned (protected seeded raw columns): {_protected}."
                    )
                return "Attributes pruned successfully."
            except Exception as e:
                return _report_tool_error(
                    tool_name="attribute_pruning_tool",
                    message=f"Error pruning attributes: {e}",
                    exception=e,
                )

        async def _run_agent_with_retries(
            *,
            agent_name: str,
            instructions: str,
            tools: list[Any],
            initial_input: str,
            session_name: str,
            set_phase: str | None = None,
        ) -> str:

            # E9f set_phased: shared session, neutral system prompt, ASSESS/PRUNE phase
            # message on the first attempt; the S/E input filter (A16-T7). Full: unchanged.
            _set_first_message = None
            if shared_session is not None:
                _set_first_message = _set_phased.phase_message(
                    set_phase, instructions, initial_input
                )
                instructions = shared_session.system_prompt
                session_name = shared_session.session_name
                shared_session.enter(set_phase)
                if self._tool_error_reporter is not None:
                    self._tool_error_reporter.phase = set_phase
            instrumentation.set_stage(
                "assessment" if "assess" in agent_name.lower() else "pruning", session_name
            )
            # E9e: drops search only under ROGUE_ONE_KNOWLEDGE_BACKEND=off (A13 edit 6).
            tools = knowledge_backend.role_tools(tools)
            client = create_openai_client_for_vllm(
                base_url=llm_cfg.endpoint_for("tester"),
                api_key=llm_cfg.api_key,
                agent_name="TesterAgent",
                organization=os.getenv("WANDB_ORGANIZATION", "rogueone"),
                project=os.getenv("WANDB_PROJECT_NAME", "rogueone_project"),
                webhook_secret=None,
            )

            agent = Agent(
                model_settings=_build_tester_model_settings(test_agent_cfg),
                name=agent_name,
                instructions=instructions,
                model=OpenAIChatCompletionsModel(
                    model=llm_cfg.tester_model,
                    openai_client=client,
                ),
                tools=tools,
            )

            def _strip_reasoning(data: CallModelData) -> ModelInputData:
                """Strip reasoning history items for vLLM chat-completions compatibility."""
                filtered = [
                    item
                    for item in data.model_data.input
                    if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]
                return ModelInputData(
                    input=filtered, instructions=data.model_data.instructions
                )

            if shared_session is None:
                _run_config = RunConfig(call_model_input_filter=_strip_reasoning)
                session = SQLiteSession(session_name)
            else:
                _run_config = RunConfig(
                    call_model_input_filter=_topology.strip_reasoning_and_special_tokens
                )
                session = shared_session.session

            dump_prompt(
                "tester", agent_name, step, instructions,
                initial_input if shared_session is None else _set_first_message,
                tools=tools, output_dir=self.cfg_experiment.output_dir,
            )
            for _attempt in range(20):
                _log(
                    f"_run_agent_with_retries: {agent_name} attempt "
                    f"{_attempt + 1}/20 (session={session_name})"
                )
                instrumentation.count("runner_runs_tester")
                try:

                    out = await Runner.run(
                        agent,
                        initial_input
                        if shared_session is None
                        else shared_session.input_for(
                            set_phase, _set_first_message, initial_input
                        ),
                        max_turns=test_agent_cfg.max_iterations,
                        session=session,
                        run_config=_run_config,
                    )
                    if len(out.final_output) > 0:
                        _log(
                            f"_run_agent_with_retries: {agent_name} returned output "
                            f"({len(out.final_output)} chars) on attempt {_attempt + 1}"
                        )
                        return out.final_output
                    _log(
                        f"_run_agent_with_retries: {agent_name} attempt "
                        f"{_attempt + 1} produced empty output; retrying"
                    )

                except Exception as e:
                    self._tool_call_counter_failed["runner"] += 1
                    if shared_session is not None:  # E9f: off-phase tool call counter
                        shared_session.note_runner_exception(set_phase, e)
                    if self._tool_error_reporter is not None:
                        self._tool_error_reporter.report(
                            tool_name="runner",
                            message=f"Error during {agent_name} run: {e}",
                            exception=e,
                            step=step,
                            context={
                                "agent_name": agent_name,
                                "session_name": session_name,
                            },
                        )
                    ConsoleManager.console_error_print(
                        f"Error during {agent_name} run "
                        f"(attempt {_attempt + 1}/20): {e}"
                    )
                    continue

            _log(
                f"_run_agent_with_retries: {agent_name} exhausted all 20 attempts "
                "without producing output."
            )
            return ""

        async def generate_assessment() -> str:
            self._assessment_stage_status = "completed"

            instructions = f"""            
            # Global Goal:
            {AgentRoles.GLOBAL_GOAL.value}
            
            
            # The Setup:
            You are part of a team of agents working together to generate features with high predictive power.
            The other agents in the team are:
            - The Scientist Agent: {AgentRoles.SCIENTIST.value}
            - The Extractor Agent: {AgentRoles.EXTRACTOR.value}

            You all work in a loop where the Scientist Agent generates focus areas, the Extractor Agent extracts attributes, and you assess the features.

            # Role
            You are the Tester Agent. Your current stage is FEATURE ASSESSMENT ONLY.
            You will assess the quality of the current feature set produced by the extractor using the pre-computed metrics as primary evidence.
            The assessment report you produce will be used by the Scientist Agent to generate new focus areas, the assessment must therefore give a clear and actionable summary.

            # Hard Constraints
            - Do not prune features in this stage.
            - Use pre-computed metrics as primary evidence.
            - Supplemental experiments are optional and only for ambiguous features.
            - Use training data only.

            # Available Resources
            - The generic_python_executor_tool environment includes a fitted XGBoost model (`assessment_model`)
              trained on the current feature set. You can use this model to:
              - Extract feature importances (e.g. `assessment_model.feature_importances_`)
              - Compute SHAP values for deeper feature-level explanations
              - Generate predictions and residual analyses
              - Perform partial dependence or interaction analyses
              Use the model when pre-computed metrics alone are ambiguous or insufficient.

            # Mandatory Workflow
            1. Call get_precomputed_metrics_tool first.
            2. Review high/low-quality features from the metrics.
            3. Optionally run targeted experiments with generic_python_executor_tool (you have access to the trained assessment_model).
            4. Take notes with take_note_tool.
            5. Return a markdown assessment report with per-feature findings and explicit evidence.

            # Output Format
            Return markdown with sections:
            - Summary
            - Feature-by-feature assessment
            - Risks and uncertainties
            - Recommended candidates for pruning (recommendations only, no pruning action)

            # Task context and optimization goal
            {self.cfg_experiment.context_and_goal}
            """

            initial_input = (
                "Assess the current feature set based on pre-computed metrics. "
                "Use this metric table as the baseline:\n\n"
                f"{metric_table_markdown}"
            )

            assessment = await _run_agent_with_retries(
                agent_name="Tester Agent (Assessment)",
                instructions=instructions,
                tools=[
                    get_precomputed_metrics_tool,
                    generic_python_executor_tool,
                    take_note_tool,
                    get_notes_tool,
                    attribute_lookup_tool,
                    search_in_literature_tool,
                ],
                initial_input=initial_input,
                session_name=f"tester_agent_assessment_step_{step or 0}",
                set_phase=_topology.PHASE_TA,
            )

            if not assessment:
                instrumentation.count("fallback_tester_report")
                if shared_session is not None:  # E9f: fallbacks are counted per phase
                    shared_session.note_fallback(_topology.PHASE_TA, "fallback_report")
                self._assessment_stage_status = "failed"
                assessment = (
                    "# Feature Assessment\n\n"
                    "Assessment agent did not return output; fallback report generated."
                )

            self._write_assessment_report(assessment_report_path, assessment)
            self._assessment_report_produced = True
            instrumentation.count("tester_assessments")
            return assessment

        async def generate_pruning(assessment_markdown: str) -> str:
            self._pruning_stage_status = "completed"

            status_before = df_attribute_explanations["Status"].copy()

            # Lookups for floor enforcement (Fix #3): prefer keeping the highest-signal
            # features when restoration is needed.
            _shap_by_feature = {
                p.feature_name: float(p.mean_abs_shap)
                for p in metric_profiles
                if p.mean_abs_shap is not None
                and not (
                    isinstance(p.mean_abs_shap, float) and np.isnan(p.mean_abs_shap)
                )
            }
            _mi_by_feature = {
                p.feature_name: float(p.mutual_information)
                for p in metric_profiles
                if p.mutual_information is not None
                and not (
                    isinstance(p.mutual_information, float)
                    and np.isnan(p.mutual_information)
                )
            }

            instructions = f"""
            # Role
            You are the Tester Agent. Your current stage is PRUNING ONLY.

            # Hard Constraints
            - You MUST use the provided assessment markdown as primary context.
            - You MUST rely primarily on pre-computed metrics.
            - Supplemental experiments are allowed only when evidence is inconclusive.
            - Use attribute_pruning_tool to prune features.
            - You MUST NOT un-prune any feature in this stage.
            - Each pruning decision must include rationale.
            - VIF >= 1e10 indicates a perfect colinearity cluster (typically the dummy-variable
              trap, or a feature paired with its `_sq` / `_x_` derivative). When you observe
              VIF >= 1e10:
                1. Group the features that share this near-infinite VIF cluster (typically the
                   original dummies of a categorical, or `feat`, `feat_sq`, `feat_x_other`).
                2. Prune ONLY the lowest-mean_abs_SHAP member of the cluster (or the lowest
                   mutual_information if SHAP is tied).
                3. NEVER prune a feature with mean_abs_SHAP > 0.1 just because of high VIF —
                   high SHAP means the feature is genuinely predictive and the high VIF is a
                   mathematical artifact, not a signal-quality problem.
            - For VIF in [10, 1e10] (high but finite), use the existing rule: prune only if
              PFI ~= 0 AND mean_abs_SHAP < 0.1.
            - Squares (`feat_sq`) and interactions (`feat_a_x_feat_b`) of features that are
              themselves active should be pruned BEFORE their bases when colinearity makes a
              choice necessary.

            # Mandatory Workflow
            1. Read pre-computed metrics with get_precomputed_metrics_tool.
            2. Read notes with get_notes_tool.
            3. Evaluate prune vs keep per feature.
            4. Call attribute_pruning_tool for prune decisions.
            5. Return a concise markdown summary with rationale and evidence source.

            # Context
            {self.cfg_experiment.context_and_goal}
            """

            initial_input = (
                "Use this assessment markdown as input context and decide what to prune:\n\n"
                f"{assessment_markdown}\n\n"
                "Pre-computed metric table:\n\n"
                f"{metric_table_markdown}"
            )

            pruning_summary = await _run_agent_with_retries(
                agent_name="Tester Agent (Pruning)",
                instructions=instructions,
                tools=[
                    get_precomputed_metrics_tool,
                    attribute_pruning_tool,
                    attribute_lookup_tool,
                    get_notes_tool,
                    generic_python_executor_tool,
                ],
                initial_input=initial_input,
                session_name=f"tester_agent_pruning_step_{step or 0}",
                set_phase=_topology.PHASE_TB,
            )

            if not pruning_summary:
                self._pruning_stage_status = "failed"
                if shared_session is not None:  # E9f: fallbacks are counted per phase
                    shared_session.note_fallback(_topology.PHASE_TB, "no_pruning_summary")
                pruning_summary = "No pruning summary produced by pruning stage."

            newly_pruned = df_attribute_explanations[
                (status_before != "Pruned")
                & (df_attribute_explanations["Status"] == "Pruned")
            ]["Attribute"].tolist()

            if newly_pruned:
                prune_mask = df_attribute_explanations["Attribute"].isin(newly_pruned)
                df_attribute_explanations.loc[prune_mask, "Prune Rationale"] = (
                    pruning_summary
                )
                df_attribute_explanations.loc[prune_mask, "Pruned Iteration"] = (
                    f"Trial {trial_index if trial_index is not None else (step if step is not None else 0)}"
                )

            active_after = int((df_attribute_explanations["Status"] != "Pruned").sum())

            # Fix #3: minimum-features floor — a near-emptied pool (e.g. 1 surviving
            # feature) is functionally identical to a complete wipe for the next
            # iteration. Restore the highest-signal newly-pruned features until at
            # least max(3, ceil(0.2 * pre_active)) features remain active.
            newly_pruned, restored = _enforce_pruning_floor(
                df_attribute_explanations=df_attribute_explanations,
                status_before=status_before,
                newly_pruned=newly_pruned,
                shap_lookup=_shap_by_feature,
                mi_lookup=_mi_by_feature,
            )
            self._fx_floor_restored = list(restored)  # E9d A− section 4 (bookkeeping only)
            if restored:
                instrumentation.count("floor_activations")
                instrumentation.count("floor_restored_features", len(restored))
                self._pruning_stage_status = "floor_enforced"
                self._num_features_pruned = max(
                    0, int(self._num_features_pruned) - len(restored)
                )
                ConsoleManager.console_print(
                    f"[yellow]Pruning floor enforced: restored {len(restored)} "
                    f"feature(s) (highest-signal newly-pruned features kept active).[/yellow]"
                )
                active_after = int(
                    (df_attribute_explanations["Status"] != "Pruned").sum()
                )

            if active_after == 0:
                df_attribute_explanations.loc[:, "Status"] = status_before
                self._pruning_stage_status = "skipped"
                ConsoleManager.console_error_print(
                    "Pruning skipped because all remaining features would be pruned."
                )
                self._pruning_decisions = []
                return "Pruning skipped: minimum feature floor reached."

            self._pruning_decisions = [
                {
                    "feature": feature,
                    "decision": "prune",
                    "evidence_source": "precomputed_metrics",
                    "rationale": pruning_summary[:500],
                }
                for feature in newly_pruned
            ]

            return pruning_summary

        def get_regression_metrics() -> TestResultRegression:
            """Evaluate features on a regression task using XGBoost with K-fold validation.

            Trains separate models for each fold, computes MSE, MAE, R2, and handles
            fold retry logic for failed trials.

            Returns:
                TestResultRegression: Object containing fold metrics and overall statistics.
            """
            base_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "random_state": self.fold_seed,
                "device": "cpu",
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "n_jobs": _XGB_N_JOBS,
            }
            mses = []
            maes = []
            r2s = []
            fold_metrics: list[dict[str, Any]] = []
            inner_cv_rmse: float | None = None
            train_r2_value: float | None = None

            if X_test is None or y_test is None:
                raise ValueError(
                    "Tabular test fold is unavailable for regression evaluation."
                )

            for attempt in range(self.fold_retry_limit + 1):
                try:
                    X_train_model = np.nan_to_num(
                        np.asarray(X_train, dtype=np.float32),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    )
                    X_test_model = np.nan_to_num(
                        np.asarray(X_test, dtype=np.float32),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    )

                    model = xgb.XGBRegressor(**base_params)
                    model.fit(X_train_model, y_train)
                    y_train_pred = model.predict(X_train_model)
                    y_pred = model.predict(X_test_model)

                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    train_r2 = r2_score(y_train, y_train_pred)
                    fold_mse = mean_squared_error(y_test, y_pred)
                    fold_mae = mean_absolute_error(y_test, y_pred)
                    fold_r2 = r2_score(y_test, y_pred)

                    # Nested-CV selection signal: score this iteration's feature set
                    # inside the training portion only, so choosing which iteration to
                    # report never consults the test fold.
                    inner_cv_rmse = _inner_cv_score(
                        X_train=X_train_model,
                        y_train=y_train,
                        base_params=base_params,
                        task_type="regression",
                        seed=self.fold_seed,
                    )
                    train_r2_value = float(train_r2)
                    l1_keys: dict[str, Any] = {}
                    if rouge_one_cfg.scientist_signal == _sci_signal.INNER_CV_L1:
                        # E14: the Scientist's scalar, on a split disjoint from the
                        # selection split above (raises L1SplitLeakageError).
                        l1_keys = _sci_signal.compute_l1_for_tester(
                            X_train=X_train_model,
                            y_train=y_train,
                            base_params=base_params,
                            task_type="regression",
                            seed=self.fold_seed,
                            balance_weights=False,
                            train_ids=l1_train_ids,
                            test_ids=l1_test_ids,
                        )

                    mses.append(fold_mse)
                    maes.append(fold_mae)
                    r2s.append(fold_r2)
                    fold_metrics.append(
                        {
                            "fold_id": selected_test_fold_id,
                            "attempt": attempt + 1,
                            "status": "success",
                            "inner_cv_RMSE": inner_cv_rmse,
                            "train_MSE": train_mse,
                            "train_MAE": train_mae,
                            "train_R2": train_r2,
                            "MSE": fold_mse,
                            "MAE": fold_mae,
                            "R2": fold_r2,
                            **l1_keys,
                        }
                    )
                    break
                except _sci_signal.L1SplitLeakageError:
                    raise
                except Exception as exc:
                    fold_metrics.append(
                        {
                            "fold_id": selected_test_fold_id,
                            "attempt": attempt + 1,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    if attempt >= self.fold_retry_limit:
                        last_error = (
                            fold_metrics[-1].get("error") if fold_metrics else "unknown"
                        )
                        return TestResultRegression(
                            mses=mses,
                            maes=maes,
                            r2s=r2s,
                            max_val=df_attributes["target"].max(),
                            min_val=df_attributes["target"].min(),
                            mean_val=df_attributes["target"].mean(),
                            report=(
                                "Fold evaluation failed after retries; "
                                f"recording NaN metrics for fold {selected_test_fold_id}. "
                                f"Last error: {last_error}"
                            ),
                            status="completed",
                            dataset_id=self.cfg_experiment.experiment_name,
                            fold_seed=self.fold_seed,
                            fold_metrics=fold_metrics,
                            aborted_fold_id=None,
                            skip_reason=None,
                        )

            return TestResultRegression(
                mses=mses,
                maes=maes,
                r2s=r2s,
                max_val=df_attributes["target"].max(),
                min_val=df_attributes["target"].min(),
                mean_val=df_attributes["target"].mean(),
                status="completed",
                dataset_id=self.cfg_experiment.experiment_name,
                fold_seed=self.fold_seed,
                fold_metrics=fold_metrics,
                aborted_fold_id=None,
                skip_reason=None,
                inner_cv_RMSE=inner_cv_rmse,
                train_R2=train_r2_value,
            )

        def get_classification_metrics_tabular() -> TestResultClassification:
            """Evaluate features on a classification task using XGBoost with K-fold validation.

            Trains separate models for each fold, computes accuracy, precision, recall, F1,
            and AUROC. Applies class balancing if configured. Handles fold retry logic for
            failed trials.

            Returns:
                TestResultClassification: Object containing fold metrics and overall statistics.
            """
            base_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": self.fold_seed,
                "device": "cpu",
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "n_jobs": _XGB_N_JOBS,
            }

            accs = []
            pres = []
            recs = []
            f1s = []
            aurocs = []
            fold_metrics: list[dict[str, Any]] = []
            inner_cv_accuracy: float | None = None
            train_accuracy_value: float | None = None

            if X_test is None or y_test is None:
                raise ValueError(
                    "Tabular test fold is unavailable for classification evaluation."
                )

            for attempt in range(self.fold_retry_limit + 1):
                try:
                    X_train_model = np.nan_to_num(
                        np.asarray(X_train, dtype=np.float32),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    )
                    X_test_model = np.nan_to_num(
                        np.asarray(X_test, dtype=np.float32),
                        nan=0.0,
                        posinf=1e6,
                        neginf=-1e6,
                    )

                    model = xgb.XGBClassifier(**base_params)

                    # Encode labels to integers for XGBoost.
                    le = LabelEncoder()
                    y_all = np.concatenate([y_train, y_test])
                    le.fit(y_all)
                    y_train_enc = np.asarray(le.transform(y_train)).ravel().astype(int)
                    y_test_enc = np.asarray(le.transform(y_test)).ravel().astype(int)

                    sample_weights_train = (
                        class_weight.compute_sample_weight("balanced", y=y_train_enc)
                        if rouge_one_cfg.do_balancing
                        else None
                    )
                    model.fit(
                        X_train_model,
                        y_train_enc,
                        sample_weight=sample_weights_train,
                    )

                    train_predictions_enc = model.predict(X_train_model)
                    predictions_enc = model.predict(X_test_model)
                    sample_weights_test = (
                        class_weight.compute_sample_weight("balanced", y=y_test_enc)
                        if rouge_one_cfg.do_balancing
                        else None
                    )
                    train_sample_weights = (
                        class_weight.compute_sample_weight("balanced", y=y_train_enc)
                        if rouge_one_cfg.do_balancing
                        else None
                    )

                    train_proba = np.asarray(model.predict_proba(X_train_model))
                    proba = np.asarray(model.predict_proba(X_test_model))

                    train_acc = accuracy_score(
                        y_train_enc,
                        train_predictions_enc,
                        sample_weight=train_sample_weights,
                    )
                    train_pre = precision_score(
                        y_train_enc,
                        train_predictions_enc,
                        sample_weight=train_sample_weights,
                        average="weighted",
                        zero_division=0,
                    )
                    train_rec = recall_score(
                        y_train_enc,
                        train_predictions_enc,
                        sample_weight=train_sample_weights,
                        average="weighted",
                        zero_division=0,
                    )
                    train_f1 = f1_score(
                        y_train_enc,
                        train_predictions_enc,
                        sample_weight=train_sample_weights,
                        average="weighted",
                        zero_division=0,
                    )

                    fold_acc = accuracy_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                    )
                    fold_pre = precision_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    )
                    fold_rec = recall_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    )
                    fold_f1 = f1_score(
                        y_test_enc,
                        predictions_enc,
                        sample_weight=sample_weights_test,
                        average="weighted",
                    )

                    classes_enc = np.unique(np.concatenate([y_train_enc, y_test_enc]))
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        try:
                            pos_idx = list(model.classes_).index(1)
                        except Exception:
                            pos_idx = 1
                        fold_proba = proba[:, pos_idx]
                    else:
                        fold_proba = proba

                    if train_proba.ndim == 2 and train_proba.shape[1] == 2:
                        try:
                            train_pos_idx = list(model.classes_).index(1)
                        except Exception:
                            train_pos_idx = 1
                        train_fold_proba = train_proba[:, train_pos_idx]
                    else:
                        train_fold_proba = train_proba

                    auroc_warning: str | None = None
                    try:
                        if len(classes_enc) == 2:
                            train_auroc = roc_auc_score(y_train_enc, train_fold_proba)
                            fold_auroc = roc_auc_score(y_test_enc, fold_proba)
                        else:
                            y_train_binarized = label_binarize(
                                y_train_enc, classes=classes_enc
                            )
                            y_test_binarized = label_binarize(
                                y_test_enc, classes=classes_enc
                            )
                            train_auroc = roc_auc_score(
                                y_train_binarized,
                                train_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                            fold_auroc = roc_auc_score(
                                y_test_binarized,
                                proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                    except Exception as auroc_exc:
                        train_auroc = float("nan")
                        fold_auroc = float("nan")
                        auroc_warning = str(auroc_exc)

                    # Nested-CV selection signal: score this iteration's feature set
                    # inside the training portion only, so choosing which iteration to
                    # report never consults the test fold.
                    inner_cv_accuracy = _inner_cv_score(
                        X_train=X_train_model,
                        y_train=y_train_enc,
                        base_params=base_params,
                        task_type="classification",
                        seed=self.fold_seed,
                        balance_weights=bool(rouge_one_cfg.do_balancing),
                    )
                    train_accuracy_value = float(train_acc)
                    l1_keys: dict[str, Any] = {}
                    if rouge_one_cfg.scientist_signal == _sci_signal.INNER_CV_L1:
                        # E14: the Scientist's scalar, on a split disjoint from the
                        # selection split above (raises L1SplitLeakageError).
                        l1_keys = _sci_signal.compute_l1_for_tester(
                            X_train=X_train_model,
                            y_train=y_train_enc,
                            base_params=base_params,
                            task_type="classification",
                            seed=self.fold_seed,
                            balance_weights=bool(rouge_one_cfg.do_balancing),
                            train_ids=l1_train_ids,
                            test_ids=l1_test_ids,
                        )

                    accs.append(fold_acc)
                    pres.append(fold_pre)
                    recs.append(fold_rec)
                    f1s.append(fold_f1)
                    aurocs.append(float(fold_auroc))
                    fold_metric_row = {
                        "fold_id": selected_test_fold_id,
                        "attempt": attempt + 1,
                        "status": "success",
                        "inner_cv_accuracy": inner_cv_accuracy,
                        "train_accuracy": train_acc,
                        "train_precision": train_pre,
                        "train_recall": train_rec,
                        "train_f1": train_f1,
                        "train_AUROC": float(train_auroc),
                        "accuracy": fold_acc,
                        "precision": fold_pre,
                        "recall": fold_rec,
                        "f1": fold_f1,
                        "AUROC": float(fold_auroc),
                    }
                    if auroc_warning is not None:
                        fold_metric_row["auroc_warning"] = auroc_warning
                    fold_metric_row.update(l1_keys)
                    fold_metrics.append(fold_metric_row)
                    break
                except _sci_signal.L1SplitLeakageError:
                    raise
                except Exception as exc:
                    fold_metrics.append(
                        {
                            "fold_id": selected_test_fold_id,
                            "attempt": attempt + 1,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    if attempt >= self.fold_retry_limit:
                        last_error = (
                            fold_metrics[-1].get("error") if fold_metrics else "unknown"
                        )
                        return TestResultClassification(
                            accuracys=accs,
                            precisions=pres,
                            recalls=recs,
                            f1s=f1s,
                            aurocs=aurocs,
                            report=(
                                "Fold evaluation failed after retries; "
                                f"recording NaN metrics for fold {selected_test_fold_id}. "
                                f"Last error: {last_error}"
                            ),
                            status="completed",
                            dataset_id=self.cfg_experiment.experiment_name,
                            fold_seed=self.fold_seed,
                            fold_metrics=fold_metrics,
                            aborted_fold_id=None,
                            skip_reason=None,
                        )

            results = pd.DataFrame(
                {
                    "fold": [
                        f"Fold {selected_test_fold_id + 1}/{self.fold_count}"
                        for _ in np.arange(1, len(accs) + 1)
                    ],
                    "accuracy": accs,
                    "precision": pres,
                    "recall": recs,
                    "f1": f1s,
                    "AUROC": aurocs,
                }
            )

            avg_row = {
                "fold": "Average",
                "accuracy": float(np.mean(accs)),
                "precision": float(np.mean(pres)),
                "recall": float(np.mean(recs)),
                "f1": float(np.mean(f1s)),
                "AUROC": float(np.mean(aurocs)),
            }
            std_row = {
                "fold": "Std Dev",
                "accuracy": float(np.std(accs)),
                "precision": float(np.std(pres)),
                "recall": float(np.std(recs)),
                "f1": float(np.std(f1s)),
                "AUROC": float(np.std(aurocs)),
            }

            results = pd.concat(
                [results, pd.DataFrame([avg_row, std_row])], ignore_index=True
            )
            ConsoleManager.print_dataframe_as_table(
                results,
                title="Classification Metrics per Fold",
                style="green",
            )

            return TestResultClassification(
                accuracys=accs,
                precisions=pres,
                recalls=recs,
                f1s=f1s,
                aurocs=aurocs,
                status="completed",
                dataset_id=self.cfg_experiment.experiment_name,
                fold_seed=self.fold_seed,
                fold_metrics=fold_metrics,
                aborted_fold_id=None,
                skip_reason=None,
                inner_cv_accuracy=inner_cv_accuracy,
                train_accuracy=train_accuracy_value,
            )

        # E9d FX switches (default off; Full-v2 takes the unchanged branches below).
        _fx_status_before = (
            df_attribute_explanations["Status"].copy() if fx_switches.any_active() else None
        )
        _fx_trial_label = f"Trial {trial_index if trial_index is not None else (step if step is not None else 0)}"
        self._fx_floor_restored = []
        if self.include_report and fx_switches.assessment_template():
            # A−: template report (sections 1–3) instead of the Stage-A LLM call.
            report = fx_switches.template_assessment(
                metric_profiles,
                metric_table_markdown,
                active_count=len(active_feature_names),
                new_count=fx_switches.count_new_features(
                    df_attribute_explanations, active_feature_names, trial_index
                ),
            )
            self._assessment_report_produced = True
            self._assessment_stage_status = "template"
            instrumentation.count("fx_template_reports")
            _log("assessment stage replaced by the E9d template report (A-).")
        elif self.include_report:
            _log("starting assessment LLM stage (generate_assessment)...")
            report = await generate_assessment()
            _log(
                f"assessment LLM stage finished "
                f"(status={self._assessment_stage_status}, "
                f"report_chars={len(report)})"
            )
        else:
            # When the caller does not need a report, skip the (expensive) LLM-driven
            # assessment stage entirely. Pruning still runs and relies primarily on
            # pre-computed metrics rather than the assessment markdown.
            self._assessment_stage_status = "skipped"
            # Allow the pruning tool gate to pass without an assessment artifact,
            # otherwise attribute_pruning_tool returns an error on every call and
            # the pruning agent retries indefinitely.
            self._assessment_report_produced = True
            report = ""
            _log("assessment LLM stage skipped (include_report=False).")

        _fx_shadow = None
        if fx_switches.shadow_active():
            # J+ shadow: J−a on a copy, with Stage B's input fixed; never enters a prompt.
            _fx_shadow = fx_switches.ja_for_tester(
                metric_profiles, X_train_assessment, active_feature_names,
                df_attribute_explanations, floor_fn=_enforce_pruning_floor,
                trial_label=_fx_trial_label, simulate=True,
            )
        _fx_ja = None
        if fx_switches.pruning_rules():
            # J−a: Stage B's rules as code, then the same floor and all-pruned guard.
            _fx_ja = fx_switches.ja_for_tester(
                metric_profiles, X_train_assessment, active_feature_names,
                df_attribute_explanations, floor_fn=_enforce_pruning_floor,
                trial_label=_fx_trial_label, simulate=False,
            )
            self._pruning_stage_status = _fx_ja.stage_status
            self._num_features_pruned = len(_fx_ja.newly_pruned)
            self._pruning_decisions = _fx_ja.records
            self._fx_floor_restored = list(_fx_ja.restored)
            pruning_summary = _fx_ja.summary
        else:
            _log("starting pruning LLM stage (generate_pruning)...")
            pruning_summary = await generate_pruning(report)
        if _fx_status_before is not None:
            _fx_pre_active = fx_switches.agent_active_count(
                df_attribute_explanations, _fx_status_before
            )
            _fx_pruned_now = fx_switches.newly_pruned_since(
                df_attribute_explanations, _fx_status_before
            )
            fx_switches.note_prune_rate(_fx_pre_active, len(_fx_pruned_now))
            if _fx_shadow is not None:
                fx_switches.log_shadow(
                    trial_index=_fx_trial_label, shadow=_fx_shadow,
                    llm_pruned=_fx_pruned_now, pre_active=_fx_pre_active,
                )
            if self.include_report and fx_switches.assessment_template():
                # A− section 4 (planner only): this iteration's prunes and their source.
                _rule_of = (
                    {d.feature: d.rule for d in _fx_ja.decisions} if _fx_ja is not None else {}
                )
                report = report + fx_switches.pruned_section(
                    [(f, _rule_of.get(f, "llm")) for f in _fx_pruned_now]
                    + [(str(f), "floor-restored") for f in self._fx_floor_restored]
                )
                self._write_assessment_report(assessment_report_path, report)
        _log(
            f"pruning LLM stage finished "
            f"(status={self._pruning_stage_status}, "
            f"pruned_decisions={len(self._pruning_decisions)}, "
            f"summary_chars={len(pruning_summary)})"
        )

        df_attributes = self.prune_attributes_in_df(
            df_attributes, df_attribute_explanations
        )

        # P4: score on the post-pruning active set. The closures below read
        # X_train / y_train / X_test, so rebind them to what survived this
        # iteration's pruning (removes the one-iteration pruning lag).
        X_train, y_train, X_test = self._rebuild_after_pruning(
            df_attributes, X_test_all_features, feature_names, X_train, y_train
        )

        match self.cfg_experiment.task_type:
            case "regression":
                _log("computing final regression fold metrics...")
                results = get_regression_metrics()

            case "classification":
                _log("computing final classification fold metrics...")
                results = get_classification_metrics_tabular()
            case _:
                raise ValueError(f"Unknown task type: {self.cfg_experiment.task_type}")

        _log(f"final metrics done (status={results.status}).")

        if self.include_report:
            results.report = report
        else:
            results.report = ""

        num_active_features = max(df_attributes.shape[1] - 1, 0)

        @wandb_logging_wrapper
        def log_results_to_wandb():
            import wandb

            wandb_step = trial_index if trial_index is not None else step

            # All `wandb.log(..., step=wandb_step)` calls in this block were
            # silently dropped on CAR runs because the LLM client logs token
            # usage on every model call, which auto-advances the run step well
            # past `wandb_step` (a small per-trial integer). Switching to
            # safe_wandb_log makes the calls fall back to auto-incremented
            # step when wandb_step would regress, so the metrics actually
            # land in the run history. The `wandb_step` value still appears
            # in payload key prefixes (e.g. `feature_metrics/step_<n>/...`)
            # for downstream plotting.
            safe_wandb_log(
                {"test_results": results.to_dict(only_metrics=True)},
                prefer_step=wandb_step,
            )

            trial_payload: dict[str, Any] = {
                "trial/active_feature_count": num_active_features,
            }
            if results.status == "completed":
                match self.cfg_experiment.task_type:
                    case "classification":
                        trial_payload.update(
                            {
                                "trial/accuracy": results.accuracy,
                                "trial/precision": results.precision,
                                "trial/recall": results.recall,
                                "trial/f1": results.f1,
                                "trial/AUROC": results.auroc,
                            }
                        )
                    case "regression":
                        trial_payload.update(
                            {
                                "trial/n_RMSE": results.n_rmse,
                                "trial/RMSE": results.rmse,
                                "trial/MSE": results.mse,
                                "trial/MAE": results.mae,
                                "trial/R2": results.r2,
                            }
                        )
            safe_wandb_log(trial_payload, prefer_step=wandb_step)

            safe_wandb_log(
                {
                    "tester_stage/assessment_status": self._assessment_stage_status,
                    "tester_stage/pruning_status": self._pruning_stage_status,
                    "tester_stage/pruning_summary": pruning_summary,
                },
                prefer_step=wandb_step,
            )

            for profile in metric_profiles:
                base_key = (
                    f"feature_metrics/step_{wandb_step or 0}/{profile.feature_name}"
                )
                safe_wandb_log(
                    {
                        f"{base_key}/mutual_information": (
                            profile.mutual_information
                            if profile.mutual_information is not None
                            else float("nan")
                        ),
                        f"{base_key}/vif": (
                            profile.vif if profile.vif is not None else float("nan")
                        ),
                        f"{base_key}/pfi_mean": (
                            profile.permutation_importance_mean
                            if profile.permutation_importance_mean is not None
                            else float("nan")
                        ),
                        f"{base_key}/pfi_std": (
                            profile.permutation_importance_std
                            if profile.permutation_importance_std is not None
                            else float("nan")
                        ),
                        f"{base_key}/noise_sensitivity": (
                            profile.noise_sensitivity
                            if profile.noise_sensitivity is not None
                            else float("nan")
                        ),
                        f"{base_key}/mean_abs_shap": (
                            profile.mean_abs_shap
                            if profile.mean_abs_shap is not None
                            else float("nan")
                        ),
                    },
                    prefer_step=wandb_step,
                )

            if self._pruning_decisions:
                pruning_table = wandb.Table(
                    columns=["feature", "decision", "evidence_source", "rationale"]
                )
                for item in self._pruning_decisions:
                    pruning_table.add_data(
                        item["feature"],
                        item["decision"],
                        item["evidence_source"],
                        item["rationale"],
                    )
                safe_wandb_log(
                    {f"pruning_decisions/step_{wandb_step or 0}": pruning_table},
                    prefer_step=wandb_step,
                )

            if metric_warning_records:
                warning_table = wandb.Table(columns=["feature", "warnings"])
                for record in metric_warning_records:
                    warning_table.add_data(record["feature"], record["warnings"])
                safe_wandb_log(
                    {f"feature_metric_warnings/step_{wandb_step or 0}": warning_table},
                    prefer_step=wandb_step,
                )

            safe_wandb_log(
                {
                    "fold_evaluation/status": results.status,
                    "fold_evaluation/dataset_id": results.dataset_id,
                    "fold_evaluation/fold_seed": results.fold_seed,
                    "fold_evaluation/aborted_fold_id": results.aborted_fold_id,
                },
                prefer_step=wandb_step,
            )

            for fold_info in results.fold_metrics or []:
                fold_id = fold_info.get("fold_id", -1)
                safe_wandb_log(
                    {
                        f"fold_details/fold_{fold_id}": fold_info,
                    },
                    prefer_step=wandb_step,
                )

            if results.status == "completed":
                match self.cfg_experiment.task_type:
                    case "regression":
                        for i in range(len(results.maes)):
                            safe_wandb_log(
                                {
                                    f"fold_metrics_{i}": {
                                        "MSE": results.mses[i],
                                        "MAE": results.maes[i],
                                        "R2": results.r2s[i],
                                        "RMSE": results.rmses[i],
                                        "NRMSE": results.n_rmses[i],
                                    }
                                },
                                prefer_step=wandb_step,
                            )

                    case "classification":
                        for i in range(len(results.accuracys)):
                            safe_wandb_log(
                                {
                                    f"fold_metrics_{i}": {
                                        "accuracy": results.accuracys[i],
                                        "precision": results.precisions[i],
                                        "recall": results.recalls[i],
                                        "f1": results.f1s[i],
                                        "AUROC": results.aurocs[i],
                                    }
                                },
                                prefer_step=wandb_step,
                            )

                    case _:
                        raise ValueError(
                            f"Unknown task type: {self.cfg_experiment.task_type}"
                        )

            try:
                artifact = wandb.Artifact(
                    name=f"feature_assessment_step_{wandb_step or 0}", type="assessment"
                )
                artifact.add_file(str(assessment_report_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                ConsoleManager.console_error_print(
                    f"Failed to log assessment artifact: {e}"
                )

            safe_wandb_log(
                {"feature_count": df_attributes.shape[1] - 1},
                prefer_step=wandb_step,
            )

            safe_wandb_log(
                {
                    f"Tester_tool_usage/{tool_name}": count
                    for tool_name, count in self._tool_call_counter.items()
                },
                prefer_step=wandb_step,
            )
            safe_wandb_log(
                {
                    f"Tester_tool_usage_success/{tool_name}": count
                    for tool_name, count in self._tool_call_counter_success.items()
                },
                prefer_step=wandb_step,
            )
            safe_wandb_log(
                {
                    f"Tester_tool_usage_failed/{tool_name}": count
                    for tool_name, count in self._tool_call_counter_failed.items()
                },
                prefer_step=wandb_step,
            )

            if self._tool_error_reporter is not None:
                safe_wandb_log(
                    {
                        "Tester_tool_errors/total": self._tool_error_reporter.total_errors,
                        "Tester_tool_errors/log_path": str(
                            self._tool_error_reporter.file_path
                        ),
                    },
                    prefer_step=wandb_step,
                )

        if self.enable_wandb:
            log_results_to_wandb()

        return results, df_attribute_explanations


if __name__ == "__main__":

    from rogueone.utils.wandb_utils import init_wandb_run
    from rogueone.utils.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(
        "./tasks/suppression_rato/config.yml"
    )

    init_wandb_run(cfg)

    console_manager = ConsoleManager
    console_manager.console_rule("Tester Agent Test", color="red")

    df_patients = pd.read_csv("./tmp/df_patients.csv")

    from rogueone.utils.train_test_splitter import TrainTestSplitter
    from pathlib import Path

    splitter = TrainTestSplitter(
        labels_file_path=cfg.label_file_path,
        test_size=0.5,
        random_state=42,
    )

    patient_attributes_train, patient_attributes_test = splitter.test_train_split(
        df_patients.copy()
    )

    console_manager.console_print(
        f"Training set size: {len(patient_attributes_train)}, Test set size: {len(patient_attributes_test)}"
    )

    tester_agent = TesterAgent(cfg)

    json_path = Path("./tmp/attribute_descriptions.json")

    try:
        try:
            attribute_explanations = pd.read_json(json_path, lines=True)
        except ValueError:
            attribute_explanations = pd.read_json(json_path)

    except Exception as e:
        console_manager.console_error_print(f"Failed to load JSON {json_path}: {e}")

    for i in range(3):
        test_result = tester_agent.test_hypotheses(
            patient_attributes_train,
            patient_attributes_test,
            attribute_explanations,
            step=i,
        )

        console_manager.console_rule(f"Test Result Iteration {i+1}", color="blue")

        result_dict = test_result.as_dict.copy()
        result_dict.pop("report", None)

        console_manager.print_dict_as_table(result_dict)

        console_manager.console_print(test_result.report)
