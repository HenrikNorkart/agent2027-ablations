import os

# Cap BLAS/OMP thread pools BEFORE importing numpy/sklearn/torch to avoid
# OpenBLAS "too many memory regions" aborts on high-core machines and when
# running parallel workers (joblib/multiprocessing oversubscription).
#
# We pick 8 as a balance: setting these to 1 serializes BLAS/SHAP/PFI/XGBoost
# CPU paths (very slow), while leaving them uncapped on a >128-core box trips
# the precompiled OpenBLAS NUM_THREADS limit. 8 leaves headroom for any outer
# parallelism (joblib up to ~16 workers stays under the 128 cap).
# Override from the shell with e.g. `OPENBLAS_NUM_THREADS=4 python ...`.
for _var in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "8")

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
import requests

from git import Tree
import pandas as pd
from pathlib import Path
from rich.live import Live
from agents import set_tracing_disabled

set_tracing_disabled(True)

from rogueone.utils import instrumentation, rouge_one_cfg, task_registry
from rogueone.utils import scientist_signal
from rogueone.utils import planner_mode
from rogueone.utils import fx_switches
from rogueone.utils import topology
from rogueone.llm.prompts import set_phased
from rogueone.utils.loop_switches import RESTART_EVERY_ITERATION
from rogueone.llm.clients import reset_llm_seed_offset, set_llm_seed_offset
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.env_validation import env_as_bool
from rogueone.utils.train_test_splitter import (
    TrainTestSplitter,
    compute_entity_duplicate_groups,
    deterministic_kfold_indices,
)
from rogueone.utils.console import ConsoleManager
from rogueone.utils.fold_safe_features import (
    REPLAY_NAN_FILLS_COUNTER,
    SEEDED_COLUMN,
    build_fold_safe_entities_folds,
    raw_coverage,
    seed_raw_feature_rows,
    seeded_attribute_names,
)
from rogueone.llm.agents import (
    ExtractorAgent,
    ScientistAgent,
    TesterAgent,
)
from rogueone.dataclasses import AttributeExplanation
from rogueone.utils.wandb_utils import (
    init_wandb_run,
    log_file_artifact,
    safe_wandb_log,
    wandb_logging_wrapper,
)
from rogueone.utils.run_vllm import start_vllm_servers


def _configure_runtime_log_filters() -> None:
    """Filter known noisy third-party log lines while keeping actionable errors."""

    allow_verbose_agent_errors = os.getenv(
        "ROGUE_ONE_VERBOSE_AGENT_ERRORS", "0"
    ).lower() in {"1", "true", "yes"}

    class _DropFilteredInputNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if allow_verbose_agent_errors:
                return True

            # Some libraries attach large payloads as structured log fields.
            # Drop those records before formatting to avoid multi-megabyte logs.
            for key in record.__dict__.keys():
                if "filtered.input" in str(key):
                    return False

            try:
                message = record.getMessage()
            except Exception:
                return True
            if "filtered.input=" in message:
                return False
            if "Error getting response; filtered.input=" in message:
                return False
            if (
                "Error getting response: Error code: 400" in message
                and "unexpected tokens remaining in message header" in message
            ):
                return False
            return True

    noisy_logger_names = [
        "agents.run_internal.run_loop",
        "agents",
        "openai.agents",
        "openai.agents.tracing",
    ]

    root_logger = logging.getLogger()
    root_has_filter = any(
        isinstance(existing, _DropFilteredInputNoise)
        for existing in root_logger.filters
    )
    if not root_has_filter:
        root_logger.addFilter(_DropFilteredInputNoise())

    for logger_name in noisy_logger_names:
        logger = logging.getLogger(logger_name)
        has_filter = any(
            isinstance(existing, _DropFilteredInputNoise) for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(_DropFilteredInputNoise())


_configure_runtime_log_filters()


@dataclass
class TestResult:
    hypothesis: str
    score: float
    attributes_used: list[
        AttributeExplanation
    ]  # List of attribute names used in the test


def _count_productive_extractor_outputs(
    *,
    num_new_columns: int,
    attribute_explanations: "pd.DataFrame | None",
) -> tuple[int, int]:
    """Compute (num_restored, num_productive) from extractor return values.

    The Extractor's auto-unprune branch flips a previously pruned feature back
    to Active without changing the column count of df_entities_attributes.
    Counting productive iterations only by column delta misclassifies a
    restoration-only iteration as "did nothing" and aborts the dataset
    (CAR run znh9zoex hit this exactly: 3 successful restorations after a
    99 % accuracy trial → spurious abort). The Extractor returns one row in
    `attribute_explanations` per productive call (new append OR restore), so
    ``len(attribute_explanations) - num_new_columns`` is a reliable
    restoration count.

    Returns ``(num_restored, num_productive)``. Both are non-negative.
    """
    num_proposed = (
        int(len(attribute_explanations)) if attribute_explanations is not None else 0
    )
    num_restored = max(num_proposed - max(num_new_columns, 0), 0)
    num_productive = max(num_new_columns, 0) + num_restored
    return num_restored, num_productive


class RogueOneAgentNetwork:

    def __init__(self, cfg: ExperimentConfig, *, enable_wandb: bool | None = None):

        self.cfg = cfg
        # E9c: under ROGUE_ONE_PLANNER_MODE=oracle|misdirected, load and validate the
        # focus file before W&B or any LLM call; None under the default "llm".
        self._planner_focus_plan = planner_mode.load_focus_plan(
            rouge_one_cfg.planner_mode,
            rouge_one_cfg.planner_focus_file,
            dataset_filter=rouge_one_cfg.dataset_filter,
            k_folds=rouge_one_cfg.k_folds,
            num_iterations=rouge_one_cfg.num_iterations,
        )
        disable_wandb_env = os.getenv("ROGUE_ONE_DISABLE_WANDB", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.enable_wandb = (
            enable_wandb
            if enable_wandb is not None
            else (rouge_one_cfg.do_logging and not disable_wandb_env)
        )

        if self.enable_wandb:
            ConsoleManager.console_rule("Setting up Weights & Biases Logging")
            init_wandb_run(cfg)
        else:
            ConsoleManager.console_print(
                "[yellow]Weights & Biases logging disabled for this run.[/yellow]"
            )

        # Reset the per-dataset tool-call-errors JSONL at the start of each
        # network init so artifacts aren't a union of multiple runs that share
        # `output_dir`. Without this, when ``rogue_one_main.py`` runs the same
        # config twice (e.g. car twice for k_folds*num_iterations averaging),
        # the second run's `agent_toolcall_errors_<dataset>` artifact contains
        # errors from the first run too — see CAR v25 (m4i2r4v9) which was
        # uploaded with errors timestamped during the previous o5zyks6e run.
        try:
            errors_path = Path(self.cfg.output_dir) / "agent_toolcall_errors.jsonl"
            errors_path.parent.mkdir(parents=True, exist_ok=True)
            errors_path.write_text("", encoding="utf-8")
        except Exception as exc:
            ConsoleManager.console_print(
                f"[yellow]Could not reset tool-call error log: {exc}[/yellow]"
            )

        # E3: fresh measurement window (tokens, counters, traces, focus log) per run.
        instrumentation.configure(self.cfg.output_dir, run_name=self.cfg.experiment_name)
        # E9d: an unknown FX flag value fails here, not mid-run; stale FX logs cleared.
        fx_switches.start_run()
        self._scored_before_test: dict[str, int] | None = None
        # E14: run-level audit of the L1 split checks (W&B summary at the end).
        scientist_signal.reset_audit()

        self.df = pd.read_csv(self.cfg.feature_file_path)

        try:
            self.df_entities_attributes = pd.DataFrame(
                self.df["id"].unique(), columns=["id"]
            )
        except:
            self.df_entities_attributes = pd.DataFrame({"id": list(self.df.index)})

        self.df_test_pool = None
        self.df_scientist_feedback_pool = pd.DataFrame()

        # ── Fold-scoped state ─────────────────────────────────────────────────────
        # The four fields below are reset at the start of each new fold run by
        # _reset_fold_state().  Do NOT add cross-fold accumulators here; use
        # df_test_pool or similar structures instead.
        #   • df_entities_attributes    – entity feature columns grow during the fold
        #   • df_attributes_explanations – attribute metadata for the current fold
        #   • df_scientist_feedback_pool – scientist feedback accumulated in the fold
        #   • focus_history              – list of focus strings chosen each iteration
        # ─────────────────────────────────────────────────────────────────────────

        self.extractor_agent = ExtractorAgent(self.cfg)
        self.scientist_agent = ScientistAgent(self.cfg)
        self.tester_agent = TesterAgent(
            self.cfg,
            include_report=True,
            enable_wandb=self.enable_wandb,
        )

        self.splitter = TrainTestSplitter(
            cfg=self.cfg,
            test_size=rouge_one_cfg.test_size,
            random_state=rouge_one_cfg.fold_seed,
        )

        self.df_attributes_explanations = pd.DataFrame(
            columns=[
                "Attribute",
                "Description",
                "Pandas Command",
                "Status",
                "Added",
                "Pruned Iteration",
                "Prune Rationale",
            ]
        )

        self.focus_history = []
        self._fold_best_metric: float | None = None
        self._fold_best_trial_id: str | None = None
        self._fold_best_state_snapshot: dict[str, Any] | None = None
        self._tabular_skip_reason: str | None = None
        self._tabular_fixed_test_ids: list[set] | None = None
        self._dataset_run_status = "running"
        self._dataset_failure_reason: str | None = None

        try:
            # Optionally stratify the outer folds so the partition can be made identical
            # to a baseline that stratifies (LLM-FE uses StratifiedKFold for
            # classification). Labels must be reindexed onto the feature frame's row
            # order -- fold membership is positional, so a mismatched order would
            # silently produce a different partition rather than an error.
            stratify_labels = None
            if (
                rouge_one_cfg.stratified_folds
                and self.cfg.task_type == "classification"
            ):
                try:
                    labels_df = self.splitter.labels
                    ordered = (
                        self.df_entities_attributes[["id"]]
                        .merge(labels_df[["id", "target"]], on="id", how="left")
                    )
                    if ordered["target"].isna().any():
                        raise ValueError(
                            "some feature-frame ids have no matching label"
                        )
                    stratify_labels = ordered["target"].to_numpy()
                except Exception as exc:
                    ConsoleManager.console_error_print(
                        "Stratified folds requested but labels could not be aligned "
                        f"({type(exc).__name__}: {exc}); using unstratified KFold."
                    )
                    stratify_labels = None

            # Finding 14 (2026-08-25): cmc/diamonds both have real exact-duplicate raw
            # rows (14.3% / 1.3%), which a plain shuffled KFold has no reason to keep
            # on the same side of a fold -- letting a tree-based evaluator score a
            # "held-out" row for free by memorizing its twin in the training fold.
            # dedup_groups is a no-op for a dataset with no duplicates (see
            # deterministic_kfold_indices), so this cannot change any fold assignment
            # that was already leak-free.
            # Finding 17 (2026-08-28): this previously passed
            # `self.df_entities_attributes`, which at __init__ time holds NOTHING but
            # `id` -- features are appended later by the Extractor. Every row therefore
            # keyed to the empty string, the whole dataset collapsed into one group,
            # and `deterministic_kfold_indices` refused to build folds ("only 1
            # distinct duplicate groups for N rows"), so the orchestrator set
            # `_tabular_skip_reason` and the run finished with rc=0 and no result. That
            # silently skipped EVERY dataset run under this code, not just the two
            # Finding 14 named. Duplicate detection has to read the raw table.
            dedup_groups = compute_entity_duplicate_groups(
                self.df,
                self.df_entities_attributes["id"].tolist(),
                exclude_cols=("id",),
            )

            fold_indices = deterministic_kfold_indices(
                n_rows=len(self.df_entities_attributes),
                n_splits=rouge_one_cfg.k_folds,
                random_state=rouge_one_cfg.fold_seed,
                stratify_labels=stratify_labels,
                dedup_groups=dedup_groups,
            )
            entity_ids = self.df_entities_attributes["id"].tolist()
            self._tabular_fixed_test_ids = [
                {entity_ids[idx] for idx in test_idx} for _, test_idx in fold_indices
            ]
        except ValueError as exc:
            self._tabular_skip_reason = str(exc)

    async def update_attribute_pool(
        self,
        attribute_explanations: pd.DataFrame,
        *,
        source_agent: str,
    ):
        for _, attr in attribute_explanations.iterrows():
            incoming_status = attr.get("Status", "Active")
            # Normalize historical status typos/casing to the canonical values.
            incoming_status_str = str(incoming_status).strip().lower()
            if incoming_status_str in {"pruned", "puned"}:
                incoming_status = "Pruned"
            else:
                incoming_status = "Active"

            mask = self.df_attributes_explanations["Attribute"] == attr["Attribute"]
            if (
                mask.any()
                and incoming_status == "Pruned"
                and str(attr["Attribute"])
                in seeded_attribute_names(self.df_attributes_explanations)
            ):
                # E6: a seeded raw-column row (augment) is never set to Pruned.
                continue
            if mask.any():
                existing_status = (
                    self.df_attributes_explanations.loc[mask, "Status"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                # Only the ExtractorAgent is allowed to un-prune previously pruned features.
                if (
                    incoming_status == "Active"
                    and (existing_status == "pruned").any()
                    and source_agent != "extractor"
                ):
                    continue

                # Update status using .loc to avoid chained assignment warnings/errors
                self.df_attributes_explanations.loc[mask, "Status"] = incoming_status

                # Keep prune metadata when provided by tester; do not overwrite with empty values.
                for metadata_col in ["Pruned Iteration", "Prune Rationale"]:
                    if metadata_col in attr and pd.notna(attr[metadata_col]):
                        metadata_value = str(attr[metadata_col]).strip()
                        if metadata_value:
                            self.df_attributes_explanations.loc[mask, metadata_col] = (
                                metadata_value
                            )

            else:
                # Create a new row DataFrame from the Series and align columns with the target dataframe
                new_row_data = attr.to_dict()
                new_row_data["Status"] = incoming_status
                new_row = pd.DataFrame([new_row_data])
                # Ensure all expected columns exist in the new row (missing columns -> NA)
                new_row = new_row.reindex(
                    columns=self.df_attributes_explanations.columns
                )
                self.df_attributes_explanations = pd.concat(
                    [self.df_attributes_explanations, new_row],
                    ignore_index=True,
                )

        # E6: rows appended after seeding are agent features, not seeds.
        if SEEDED_COLUMN in self.df_attributes_explanations.columns:
            self.df_attributes_explanations[SEEDED_COLUMN] = (
                self.df_attributes_explanations[SEEDED_COLUMN]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )

        # De-fragment the dataframe
        self.df_attributes_explanations = self.df_attributes_explanations.copy()
        self.df_test_pool = (
            self.df_test_pool.copy() if self.df_test_pool is not None else None
        )

    @staticmethod
    def _build_scientist_feedback_row(
        test_results_obj: object,
        report: str,
        trial_id: str,
        signal: str | None = None,
    ) -> dict:
        """Build a scientist-safe feedback payload containing train metrics + assessment only.

        E14: under ``ROGUE_ONE_SCIENTIST_SIGNAL=inner_cv_l1`` the ``train_*`` metrics
        are replaced by the L1 scalar (``cv_accuracy`` / ``cv_RMSE``).
        """
        fold_metrics = getattr(test_results_obj, "fold_metrics", None) or []
        successful = [m for m in fold_metrics if m.get("status") == "success"]

        def _mean_metric(key: str) -> float | None:
            values = [
                float(m[key]) for m in successful if key in m and m[key] is not None
            ]
            if len(values) == 0:
                return None
            return float(sum(values) / len(values))

        signal = rouge_one_cfg.scientist_signal if signal is None else signal
        if signal == scientist_signal.INNER_CV_L1:
            return {
                "id": trial_id,
                "status": getattr(test_results_obj, "status", "completed"),
                "dataset_id": getattr(test_results_obj, "dataset_id", ""),
                "fold_seed": getattr(test_results_obj, "fold_seed", None),
                "aborted_fold_id": getattr(test_results_obj, "aborted_fold_id", None),
                "skip_reason": getattr(test_results_obj, "skip_reason", None),
                "feature_assessment": report,
                scientist_signal.L1_COLUMN_CLASSIFICATION: _mean_metric(
                    scientist_signal.L1_FOLD_KEY_CLASSIFICATION
                ),
                scientist_signal.L1_COLUMN_REGRESSION: _mean_metric(
                    scientist_signal.L1_FOLD_KEY_REGRESSION
                ),
            }

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

    def _reset_fold_state(self) -> None:
        """Reset all fold-scoped state before starting a new fold run.

        Called automatically at the start of every fold (when iteration_in_fold == 0)
        so that state from fold N cannot leak into fold N+1.

        Fields reset:
        - df_entities_attributes: rebuilt to the single-column entity-id base table
        - df_attributes_explanations: empty DataFrame with canonical columns
        - df_scientist_feedback_pool: empty DataFrame
        - focus_history: empty list
        - scientist_agent._note_book: cleared to ""; the ScientistAgent accumulates
          free-text observations in this private attribute across iterations — it must
          be cleared at fold boundaries to prevent cross-fold knowledge contamination.

        E6: under ``ROGUE_ONE_FEATURE_POOL_MODE=augment`` the entity and attribute
        tables are then re-seeded with every raw column (``seed_raw_feature_rows``) on
        EVERY call, so any later reset (E9a restarts) starts from the seeded pool too.
        Under ``replace`` (default) the reset is exactly the pre-E6 id-only reset.
        """
        try:
            self.df_entities_attributes = pd.DataFrame(
                self.df["id"].unique(), columns=["id"]
            )
        except Exception:
            self.df_entities_attributes = pd.DataFrame({"id": list(self.df.index)})

        self.df_attributes_explanations = pd.DataFrame(
            columns=[
                "Attribute",
                "Description",
                "Pandas Command",
                "Status",
                "Added",
                "Pruned Iteration",
                "Prune Rationale",
            ]
        )
        if rouge_one_cfg.feature_pool_mode == "augment":
            self.df_entities_attributes, self.df_attributes_explanations = (
                seed_raw_feature_rows(
                    self.df,
                    self.df_entities_attributes,
                    self.df_attributes_explanations,
                )
            )
        self.df_scientist_feedback_pool = pd.DataFrame()
        self.focus_history = []
        instrumentation.reset_fold_scope()
        self._fold_best_metric = None
        self._fold_best_trial_id = None
        self._fold_best_state_snapshot = None
        # ScientistAgent._note_book accumulates observations across iterations;
        # reset it here to prevent fold-to-fold knowledge leakage.
        self.scientist_agent._note_book = ""

    def _primary_metric_name(self) -> str:
        return "accuracy" if self.cfg.task_type == "classification" else "n_RMSE"

    def _is_improved_metric(self, candidate: float, current_best: float) -> bool:
        if self.cfg.task_type == "classification":
            return candidate > current_best
        return candidate < current_best

    def _capture_fold_state_snapshot(self, trial_id: str, metric_value: float) -> None:
        self._fold_best_metric = metric_value
        self._fold_best_trial_id = trial_id
        self._fold_best_state_snapshot = {
            "df_entities_attributes": self.df_entities_attributes.copy(deep=True),
            "df_attributes_explanations": self.df_attributes_explanations.copy(
                deep=True
            ),
            "scientist_note_book": getattr(self.scientist_agent, "_note_book", ""),
        }

    def _restore_fold_state_snapshot(self) -> None:
        if self._fold_best_state_snapshot is None:
            return
        self.df_entities_attributes = self._fold_best_state_snapshot[
            "df_entities_attributes"
        ].copy(deep=True)
        self.df_attributes_explanations = self._fold_best_state_snapshot[
            "df_attributes_explanations"
        ].copy(deep=True)
        self.scientist_agent._note_book = self._fold_best_state_snapshot[
            "scientist_note_book"
        ]

    def _apply_best_so_far_rollback(
        self,
        *,
        test_results_dict: dict[str, Any],
        trial_id: str,
        fold_run_id: int,
        iteration_in_fold: int,
    ) -> None:
        """Rollback fold-scoped feature state when a new iteration underperforms.

        We still keep every trial in df_test_pool for auditability. Rollback only affects
        the state used to seed the *next* iteration in the same fold.
        """
        metric_name = self._primary_metric_name()
        metric_value = pd.to_numeric(
            test_results_dict.get(metric_name), errors="coerce"
        )
        if pd.isna(metric_value):
            if self._fold_best_state_snapshot is not None:
                self._restore_fold_state_snapshot()
                ConsoleManager.console_print(
                    "[yellow]Best-so-far rollback: restored fold state after NaN metric "
                    f"on fold {fold_run_id}, iteration {iteration_in_fold}, trial {trial_id}."
                    "[/yellow]"
                )
            return

        metric_value_float = float(metric_value)
        if self._fold_best_metric is None:
            self._capture_fold_state_snapshot(trial_id, metric_value_float)
            return

        if self._is_improved_metric(metric_value_float, self._fold_best_metric):
            self._capture_fold_state_snapshot(trial_id, metric_value_float)
            return

        self._restore_fold_state_snapshot()
        ConsoleManager.console_print(
            "[yellow]Best-so-far rollback: restored fold state from trial "
            f"{self._fold_best_trial_id} (metric={self._fold_best_metric:.6g}) after "
            f"underperforming trial {trial_id} (metric={metric_value_float:.6g}) on fold "
            f"{fold_run_id}, iteration {iteration_in_fold}."
            "[/yellow]"
        )

    def _snapshot_scored(self) -> dict[str, int]:
        """Attribute -> 1 iff not ``Pruned`` right now (E3 ``Scored``, taken before the Tester)."""
        df = self.df_attributes_explanations
        return {
            str(a): int(str(st).strip().lower() != "pruned")
            for a, st in zip(df["Attribute"], df["Status"])
        }

    def _log_iteration(self, index: int, fold_run_id: int, iteration_in_fold: int, status: Any) -> None:
        extra: dict[str, Any] = {}
        try:
            df = self.df_attributes_explanations
            pruned = df["Status"].astype(str).str.strip().str.lower() == "pruned"
            extra = {
                "active_features": int((~pruned).sum()),
                "pruned_features": int(pruned.sum()),
                "pool_rows": int(len(df)),
                "entity_columns": int(max(len(self.df_entities_attributes.columns) - 1, 0)),
            }
        except Exception:
            pass
        # E6 (A03 g): share of raw columns whose seeded row is Active; 0 under replace.
        try:
            coverage = raw_coverage(self.df, self.df_attributes_explanations)
            extra["raw_coverage"] = coverage
            if self.enable_wandb:
                safe_wandb_log(
                    {
                        "raw_coverage": coverage,
                        "raw_coverage_iteration_global": index,
                        "raw_coverage_fold_run_id": fold_run_id,
                        "raw_coverage_iteration_in_fold": iteration_in_fold,
                    },
                    prefer_step=index,
                )
        except Exception:
            pass
        instrumentation.log_iteration(index, fold_run_id, iteration_in_fold, status, extra)

    async def forward_pass(self, index: int):
        # E3: tag every LLM call of this iteration (contextvar) and log the iteration.
        fold_run_id = index // rouge_one_cfg.num_iterations
        iteration_in_fold = index % rouge_one_cfg.num_iterations
        instrumentation.set_iteration(index, fold_run_id, iteration_in_fold)
        self._scored_before_test = None
        status = None
        # E9a restart seed rule: under every_iteration each restart samples with
        # llm_seed + index (all 50 draws distinct); off leaves the seed untouched.
        _seed_token = (
            set_llm_seed_offset(index)
            if rouge_one_cfg.restart_mode == RESTART_EVERY_ITERATION
            else None
        )
        try:
            status = await self._forward_pass(index)
            return status
        finally:
            if _seed_token is not None:
                reset_llm_seed_offset(_seed_token)
            # E9f: close this iteration's shared session (set_phased only; None in Full).
            _set_shared = getattr(self, "_set_shared_session", None)
            if _set_shared is not None:
                self._set_shared_session = None
                _set_shared.close(fold_run_id, iteration_in_fold)
            self._log_iteration(index, fold_run_id, iteration_in_fold, status)

    async def _forward_pass(self, index: int):
        # Derive per-fold identity: fold_run_id identifies which of the 5 folds this
        # iteration belongs to; iteration_in_fold is the 0-based position within that fold.
        fold_run_id = index // rouge_one_cfg.num_iterations
        iteration_in_fold = index % rouge_one_cfg.num_iterations

        # Reset all fold-scoped state at the start of each new fold run
        # (E9a: before every iteration under ROGUE_ONE_RESTART_MODE=every_iteration).
        if iteration_in_fold == 0 or rouge_one_cfg.restart_mode == RESTART_EVERY_ITERATION:
            self._reset_fold_state()

        # E9f (A16): under ROGUE_ONE_TOPOLOGY=set_phased one new session per iteration,
        # shared by S, every E round, T-A and T-B, with one neutral system prompt.
        # three_agent (default, Full-v2): no shared session, every call unchanged.
        _set_kw: dict = {}
        if getattr(rouge_one_cfg, "topology", topology.THREE_AGENT) == topology.SET_PHASED:
            self._set_shared_session = topology.SharedIterationSession(
                index,
                set_phased.system_prompt(
                    self.cfg.context_and_goal, set_phased.raw_columns_str(self.df.columns)
                ),
                output_dir=self.cfg.output_dir,
            )
            _set_kw = {"shared_session": self._set_shared_session}

        ######### Setting Focus of Investigation #########
        # E9d P− (ROGUE_ONE_PLANNER_MODE=none): no Scientist call; the Extractor sets
        # its own focus from a round-1 digest (None under every other planner mode).
        _fx_self_focus = fx_switches.begin_self_focus(
            self.df_scientist_feedback_pool,
            self.df_attributes_explanations,
            self.focus_history,
            index=index,
            fold_run_id=fold_run_id,
            iteration_in_fold=iteration_in_fold,
        )
        if _fx_self_focus is None:
            ConsoleManager.console_rule("Setting Focus of Investigation")
            instrumentation.set_caller("scientist")
            _t_stage = instrumentation.stage_begin("scientist")
            # E9c: a validated focus file (oracle / misdirected) replaces the Scientist.
            # None (the default, ROGUE_ONE_PLANNER_MODE=llm) keeps the Full-v2 path below.
            _focus_plan = getattr(self, "_planner_focus_plan", None)
            if _focus_plan is None:
                fx_switches.reset_scientist_memory(self.scientist_agent)  # E9d M−
                focus = await self.scientist_agent.determine_focus(
                    self.df,
                    self.df_scientist_feedback_pool,
                    self.df_attributes_explanations,
                    fx_switches.scientist_history_view(self.focus_history),  # E9d M−
                    index=index,
                    max_index=rouge_one_cfg.num_iterations,
                    iteration_in_fold=iteration_in_fold,
                    **_set_kw,
                )
                _focus_source = instrumentation.pop_focus_source(
                    os.getenv("ROGUE_ONE_PLANNER_MODE", "llm")
                )
            else:
                focus = _focus_plan.focus_for(fold_run_id, iteration_in_fold)
                _focus_source = _focus_plan.mode
            instrumentation.stage_end(_t_stage)
            instrumentation.log_focus(
                index, fold_run_id, iteration_in_fold, focus, _focus_source,
            )
            self.focus_history.append(focus)
            ConsoleManager.console_print(f"Focus of Investigation: [bold]{focus}[/bold]\n")
        else:
            focus = None  # set by the Extractor's round-1 FOCUS: line below

        ######### Extracting Attributes #########
        num_known_attributes = len(self.df_entities_attributes.columns) - 1
        ConsoleManager.console_rule("Extracting Attributes")
        ConsoleManager.progress_bar_update(description="Extracting attributes")
        instrumentation.set_caller("extractor")
        _t_stage = instrumentation.stage_begin("extractor")
        self.df_entities_attributes, _attribute_explanations = (
            await self.extractor_agent.generate_patient_attributes(
                self.df,
                self.df_entities_attributes,
                self.df_attributes_explanations,
                focus,
                step=index,
                **({"self_focus": _fx_self_focus} if _fx_self_focus is not None else {}),
                **_set_kw,
            )
        )

        instrumentation.stage_end(_t_stage)
        if _fx_self_focus is not None:
            # E9d P−: the Extractor's own FOCUS: line is this iteration's focus.
            focus = _fx_self_focus.finalize()
            instrumentation.log_focus(index, fold_run_id, iteration_in_fold, focus, "extractor_self")
            self.focus_history.append(focus)
            ConsoleManager.console_print(f"Focus of Investigation (Extractor, P−): [bold]{focus}[/bold]\n")

        ######## Logging and Updating Attribute Explanations #########
        ConsoleManager.console_rule("Extraction Summary")
        ConsoleManager.print_dataframe_as_table(
            _attribute_explanations,
            title="Extraction Summary",
        )

        # See _count_productive_extractor_outputs docstring for why we
        # cannot rely on column delta alone.
        num_new_columns = (
            len(self.df_entities_attributes.columns) - 1 - num_known_attributes
        )
        num_restored, num_productive_attributes = _count_productive_extractor_outputs(
            num_new_columns=num_new_columns,
            attribute_explanations=_attribute_explanations,
        )

        # Always log per-iteration extractor activity (success or abort) so we can
        # trace why an iteration produced 0 productive features even when the
        # JSONL log has zero error entries. Counters are reset at the start of
        # every generate_patient_attributes call, so they are per-iteration.
        self._log_extractor_iteration_activity(
            index=index,
            iteration_in_fold=iteration_in_fold,
            fold_run_id=fold_run_id,
            focus=focus,
            num_known_attributes=num_known_attributes,
            num_new_attributes=num_new_columns,
            num_restored_attributes=num_restored,
            num_productive_attributes=num_productive_attributes,
            attribute_explanations=_attribute_explanations,
        )

        if num_productive_attributes == 0:
            # Distinguish between two zero-productivity cases:
            #
            # 1. ``iteration_in_fold == 0`` — the very first iteration of a fold.
            #    df_features is empty (just the id column), so the Tester has
            #    nothing to evaluate. The fold cannot contribute any trial to
            #    fold_aggregation. ABORT the dataset.
            #
            # 2. ``iteration_in_fold > 0`` — fold already has at least one
            #    completed trial from a prior iteration. The Extractor has
            #    simply run out of fresh ideas (extremely common on small
            #    datasets like CAR with only 6 raw categorical columns; the
            #    model exhausts encodings/interactions in 1-2 iterations).
            #    Aborting the *dataset* here is too aggressive — fold
            #    aggregation only needs ONE completed trial per fold and
            #    we already have that. Skip *this iteration only* and
            #    continue to the next iteration.
            #
            # Before this change, every dataset whose feature space the model
            # exhausted before iter == num_iterations - 1 aborted spuriously
            # (CAR runs ca90miqg / p9t3rayi / wi5r0oki are textbook cases).
            failure_reason = "No new or restored attributes were produced."

            self._log_extractor_abort(
                index=index,
                iteration_in_fold=iteration_in_fold,
                fold_run_id=fold_run_id,
                focus=focus,
                num_known_attributes=num_known_attributes,
                attribute_explanations=_attribute_explanations,
                reason=failure_reason,
            )

            if iteration_in_fold > 0:
                ConsoleManager.console_print(
                    f"[yellow]Extractor produced 0 productive features at iter "
                    f"{iteration_in_fold} of fold {fold_run_id}. The fold already "
                    f"has trials from earlier iterations; skipping this "
                    f"iteration without running the Tester.[/yellow]\n"
                )
                skipped_iter_result: dict[str, Any] = {
                    "status": "iteration_skipped",
                    "dataset_id": self.cfg.experiment_name,
                    "fold_seed": rouge_one_cfg.fold_seed,
                    "skip_reason": failure_reason,
                    "report": failure_reason,
                    "id": "Trial " + str(index),
                    "test_fold_id": fold_run_id,
                    "iteration_in_fold": iteration_in_fold,
                }
                if self.df_test_pool is None:
                    self.df_test_pool = pd.DataFrame([skipped_iter_result])
                else:
                    self.df_test_pool = pd.concat(
                        [self.df_test_pool, pd.DataFrame([skipped_iter_result])],
                        ignore_index=True,
                    )
                return "iteration_skipped"

            ConsoleManager.console_print(
                "[red]No features produced on iteration 0 of this fold; "
                "the Tester has nothing to evaluate. Aborting dataset "
                "evaluation.[/red]\n"
            )

            aborted_result: dict[str, Any] = {
                "status": "aborted",
                "dataset_id": self.cfg.experiment_name,
                "fold_seed": rouge_one_cfg.fold_seed,
                "aborted_fold_id": fold_run_id,
                "skip_reason": failure_reason,
                "report": failure_reason,
                "id": "Trial " + str(index),
                "test_fold_id": fold_run_id,
            }

            if self.df_test_pool is None:
                self.df_test_pool = pd.DataFrame([aborted_result])
            else:
                self.df_test_pool = pd.concat(
                    [self.df_test_pool, pd.DataFrame([aborted_result])],
                    ignore_index=True,
                )

            scientist_feedback_row = {
                "id": aborted_result["id"],
                "status": "aborted",
                "dataset_id": self.cfg.experiment_name,
                "fold_seed": rouge_one_cfg.fold_seed,
                "aborted_fold_id": aborted_result["aborted_fold_id"],
                "skip_reason": failure_reason,
                "feature_assessment": failure_reason,
                "train_accuracy": None,
                "train_precision": None,
                "train_recall": None,
                "train_f1": None,
                "train_AUROC": None,
                "train_MSE": None,
                "train_MAE": None,
                "train_R2": None,
            }
            if rouge_one_cfg.scientist_signal == scientist_signal.INNER_CV_L1:
                # E14: same row shape as a completed trial under L1 (no train_*).
                scientist_feedback_row = {
                    k: v
                    for k, v in scientist_feedback_row.items()
                    if not k.startswith("train_")
                }
                scientist_feedback_row[scientist_signal.L1_COLUMN_CLASSIFICATION] = None
                scientist_feedback_row[scientist_signal.L1_COLUMN_REGRESSION] = None
            self.df_scientist_feedback_pool = pd.concat(
                [
                    self.df_scientist_feedback_pool,
                    pd.DataFrame([scientist_feedback_row]),
                ],
                ignore_index=True,
            )

            await self.dump_data(index)
            await self.save_report(failure_reason, index)
            return "aborted"

        await self.update_attribute_pool(
            _attribute_explanations,
            source_agent="extractor",
        )

        ConsoleManager.console_print(
            f"Extracted {num_productive_attributes} attributes "
            f"({num_new_columns} new, {num_restored} restored).\n"
        )

        ######### Splitting Patient Data #########
        ConsoleManager.console_rule("Splitting Data")
        ConsoleManager.progress_bar_update(description="Splitting data")

        try:
            if self._tabular_skip_reason is not None:
                raise ValueError(self._tabular_skip_reason)

            if not self._tabular_fixed_test_ids:
                raise ValueError("Fixed tabular folds are unavailable.")

            labels_df = self.splitter.labels.copy()
            if "id" not in labels_df.columns and "patient_id" in labels_df.columns:
                labels_df = labels_df.rename(columns={"patient_id": "id"})

            # Fold-safe recomputation (Finding 14, 2026-08-25): self.df_entities_attributes
            # is built once over the FULL dataset, so any feature whose command aggregates
            # across rows (groupby/transform, rank, rolling, ...) would otherwise fit its
            # statistic using rows from this fold's own held-out test set. Recompute every
            # attribute separately against this fold's train partition and its test
            # partition instead of slicing the globally-computed columns -- see
            # rogueone/utils/fold_safe_features.py for why this is necessary and why it is
            # sufficient (row-wise features are unaffected either way).
            df_entities_folds = build_fold_safe_entities_folds(
                df_raw=self.df,
                attribute_explanations=self.df_attributes_explanations,
                labels_df=labels_df,
                fixed_test_ids=self._tabular_fixed_test_ids,
                current_test_fold_id=fold_run_id,
            )
        except ValueError as exc:
            skip_reason = str(exc)
            ConsoleManager.console_print(
                f"[yellow]Dataset skipped for this run: {skip_reason}[/yellow]"
            )

            skipped_result = {
                "status": "skipped",
                "dataset_id": self.cfg.experiment_name,
                "fold_seed": rouge_one_cfg.fold_seed,
                "aborted_fold_id": None,
                "skip_reason": skip_reason,
                "id": "Trial " + str(index),
            }

            if self.df_test_pool is None:
                self.df_test_pool = pd.DataFrame([skipped_result])
            else:
                self.df_test_pool = pd.concat(
                    [self.df_test_pool, pd.DataFrame([skipped_result])],
                    ignore_index=True,
                )

            await self.dump_data(index)
            await self.save_report(skip_reason, index)
            return "skipped"

        ######### Generating and Testing Hypotheses #########
        ConsoleManager.console_rule("Generating and Testing Hypotheses")
        ConsoleManager.progress_bar_update(description="Testing hypotheses")
        # E3: the Tester mutates the attribute table in place, so copy ``Scored`` now.
        self._scored_before_test = self._snapshot_scored()
        instrumentation.pop_inner_cv_scores()
        instrumentation.set_caller("tester")
        _t_stage = instrumentation.stage_begin("tester")
        test_results, _attribute_explanations = await self.tester_agent.test_hypotheses(
            df_entities_folds,
            self.df_attributes_explanations,
            step=fold_run_id,
            trial_index=index,
            **_set_kw,
        )
        instrumentation.stage_end(_t_stage)
        await self.update_attribute_pool(
            _attribute_explanations,
            source_agent="tester",
        )
        trial_id = "Trial " + str(index)
        test_results_dict = test_results.as_dict.copy()
        test_results_dict["test_fold_id"] = fold_run_id
        test_results_dict["iteration_in_fold"] = iteration_in_fold
        test_results_dict["id"] = trial_id
        # E3 (A21 data contract): the inner-CV fold scores whose mean is inner_cv_*.
        _inner_scores = instrumentation.pop_inner_cv_scores()
        test_results_dict["inner_cv_fold_scores"] = (
            json.dumps(_inner_scores) if _inner_scores is not None else None
        )

        if rouge_one_cfg.scientist_signal == scientist_signal.INNER_CV_L1:
            # E14: the L1 scalar the Scientist is shown, kept in the test pool for analysis.
            _l1_row = self._build_scientist_feedback_row(
                test_results, "", trial_id, signal=scientist_signal.INNER_CV_L1
            )
            test_results_dict["l1_cv_accuracy"] = _l1_row[
                scientist_signal.L1_COLUMN_CLASSIFICATION
            ]
            test_results_dict["l1_cv_RMSE"] = _l1_row[
                scientist_signal.L1_COLUMN_REGRESSION
            ]

        if rouge_one_cfg.enable_rollback:
            self._apply_best_so_far_rollback(
                test_results_dict=test_results_dict,
                trial_id=trial_id,
                fold_run_id=fold_run_id,
                iteration_in_fold=iteration_in_fold,
            )

        if self.df_test_pool is None:
            self.df_test_pool = pd.DataFrame([test_results_dict])
        else:
            self.df_test_pool = pd.concat(
                [self.df_test_pool, pd.DataFrame([test_results_dict])],
                ignore_index=True,
            )

        report = test_results_dict.pop("report")

        scientist_feedback_row = self._build_scientist_feedback_row(
            test_results,
            report,
            trial_id,
        )
        self.df_scientist_feedback_pool = pd.concat(
            [self.df_scientist_feedback_pool, pd.DataFrame([scientist_feedback_row])],
            ignore_index=True,
        )

        await self.dump_data(index)
        await self.save_report(report, index)
        ConsoleManager.print_dict_as_table(test_results_dict)
        ConsoleManager.console_print(report)
        return test_results_dict.get("status", "completed")

    async def dump_data(self, step: int):
        dst_dir = self.cfg.output_dir / "intermediate_steps"
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)

        self.df_entities_attributes.to_csv(
            dst_dir / f"df_entities_attributes_trail_{step}.csv",
            index=False,
        )
        self._dump_trial_tables(step, dst_dir)

    def _dump_trial_tables(self, step: int, dst_dir: Path) -> None:
        """E3 per-trial tables: attribute snapshot (A10-T18, A18-T2, A22-T1) and S's feedback pool."""
        try:
            df = self.df_attributes_explanations.copy()
            scored = getattr(self, "_scored_before_test", None)
            tester_ran = scored is not None
            status_active = (df["Status"].astype(str).str.strip().str.lower() != "pruned").astype(int)
            fallback_names = instrumentation.fallback_feature_names()
            df["Fallback"] = df["Attribute"].astype(str).isin(fallback_names).astype(int)
            if "Seeded" not in df.columns:
                df["Seeded"] = 0  # E6 overwrites this column when the augment mode seeds rows
            else:
                df["Seeded"] = (
                    df["Seeded"].astype("boolean").fillna(False).astype(int)
                )
            # Scored = not Pruned before test_hypotheses (spec); Scored_post_prune = not
            # Pruned after it, i.e. the set the P4 fix (E2) actually scores.
            df["Scored"] = (
                df["Attribute"].astype(str).map(lambda a: int(scored.get(a, 0))) if tester_ran else 0
            )
            df["Scored_post_prune"] = status_active if tester_ran else 0
            index = step
            df["Trial"] = f"Trial {index}"
            df["fold_run_id"] = index // rouge_one_cfg.num_iterations
            df["iteration_in_fold"] = index % rouge_one_cfg.num_iterations
            df.to_csv(dst_dir / f"df_attributes_explanations_trail_{step}.csv", index=False)
            self.df_scientist_feedback_pool.to_csv(
                dst_dir / f"df_scientist_feedback_trail_{step}.csv", index=False
            )
        except Exception as exc:  # instrumentation must not break the run
            ConsoleManager.console_print(f"[yellow]Could not write trial tables: {exc}[/yellow]")

    async def save_report(self, report: str | None, index: int):
        if report is None:
            return
        dir = self.cfg.output_dir / "tester_reports"
        if not dir.exists():
            dir.mkdir(parents=True)
        dst_path = dir / f"tester_report_iteration_{index}.md"
        with open(dst_path, "w") as f:
            f.write(report)

    def _extractor_counter_snapshot(self) -> dict[str, int]:
        """Snapshot the Extractor's per-iteration tool-call counters.

        Counters are reset at the start of every ``generate_patient_attributes`` call,
        so reading them right after that call returns gives per-iteration values.
        Returns zero-valued defaults when the Extractor has not run yet.
        """
        agent = getattr(self, "extractor_agent", None)
        if agent is None:
            return {
                "append_total": 0,
                "append_success": 0,
                "append_failed": 0,
                "pandas_total": 0,
                "pandas_success": 0,
                "pandas_failed": 0,
            }
        return {
            "append_total": int(
                getattr(agent, "_tool_call_counter", {}).get("append_new_attribute", 0)
            ),
            "append_success": int(
                getattr(agent, "_tool_call_counter_success", {}).get(
                    "append_new_attribute", 0
                )
            ),
            "append_failed": int(
                getattr(agent, "_tool_call_counter_failed", {}).get(
                    "append_new_attribute", 0
                )
            ),
            "pandas_total": int(
                getattr(agent, "_tool_call_counter", {}).get("generic_pandas_tool", 0)
            ),
            "pandas_success": int(
                getattr(agent, "_tool_call_counter_success", {}).get(
                    "generic_pandas_tool", 0
                )
            ),
            "pandas_failed": int(
                getattr(agent, "_tool_call_counter_failed", {}).get(
                    "generic_pandas_tool", 0
                )
            ),
        }

    def _log_extractor_iteration_activity(
        self,
        *,
        index: int,
        iteration_in_fold: int,
        fold_run_id: int,
        focus: str,
        num_known_attributes: int,
        num_new_attributes: int,
        num_restored_attributes: int = 0,
        num_productive_attributes: int | None = None,
        attribute_explanations: pd.DataFrame | None = None,
    ) -> None:
        """Log per-iteration Extractor activity to W&B.

        Always called after the Extractor runs, regardless of whether the iteration
        produced new attributes. Intentionally captures success **and** failure counts
        so we can tell apart "Extractor never tried" from "Extractor tried hard and
        every attempt was rejected" — the JSONL error log only records failures, so
        without this we cannot diagnose iterations that abort with 0 logged errors.

        ``num_new_attributes`` is the *column-count delta* (new appends only).
        ``num_restored_attributes`` counts auto-unpruned features (no column
        delta but still productive work). ``num_productive_attributes`` is the
        sum and is what the abort gate actually checks; logging all three
        makes restoration-only iterations distinguishable in W&B (CAR run
        znh9zoex would have shown new=0, restored=3, productive=3).
        """
        counters = self._extractor_counter_snapshot()
        proposals = (
            int(len(attribute_explanations)) if attribute_explanations is not None else 0
        )
        if num_productive_attributes is None:
            num_productive_attributes = num_new_attributes + num_restored_attributes

        ConsoleManager.console_print(
            f"[dim]extractor_iteration[/dim] iter={index} fold={fold_run_id}/"
            f"{iteration_in_fold} new={num_new_attributes} "
            f"restored={num_restored_attributes} productive={num_productive_attributes} "
            f"append={counters['append_total']} "
            f"(ok={counters['append_success']}, fail={counters['append_failed']}) "
            f"pandas={counters['pandas_total']} "
            f"(fail={counters['pandas_failed']}) "
            f"proposals={proposals}"
        )

        if not self.enable_wandb:
            return

        # NOTE: do not pass step=index — wandb's step axis is auto-advanced by
        # the per-LLM-call token logging in clients.py, so any small iteration
        # index gets silently dropped by wandb. iteration_global is in the
        # payload itself, so plots can pivot off that field instead.
        safe_wandb_log(
            {
                "extractor_iteration/iteration_global": index,
                "extractor_iteration/iteration_in_fold": iteration_in_fold,
                "extractor_iteration/fold_run_id": fold_run_id,
                "extractor_iteration/active_features_before": num_known_attributes,
                "extractor_iteration/new_features_added": num_new_attributes,
                "extractor_iteration/restored_features": num_restored_attributes,
                "extractor_iteration/productive_features": num_productive_attributes,
                "extractor_iteration/proposed_explanations": proposals,
                "extractor_iteration/append_attempts_total": counters["append_total"],
                "extractor_iteration/append_attempts_success": counters[
                    "append_success"
                ],
                "extractor_iteration/append_attempts_failed": counters[
                    "append_failed"
                ],
                "extractor_iteration/pandas_attempts_total": counters["pandas_total"],
                "extractor_iteration/pandas_attempts_failed": counters[
                    "pandas_failed"
                ],
            }
        )

    def _log_extractor_abort(
        self,
        *,
        index: int,
        iteration_in_fold: int,
        fold_run_id: int,
        focus: str,
        num_known_attributes: int,
        attribute_explanations: pd.DataFrame | None,
        reason: str,
    ) -> None:
        """Log a structured abort event to W&B with full diagnostic context.

        Captured fields make it possible to reconstruct, from the W&B run alone,
        which iteration aborted, what the Scientist focus was, how many extractor
        attempts happened, and what the Extractor proposed (if anything) before
        giving up.
        """
        if not self.enable_wandb:
            return

        counters = self._extractor_counter_snapshot()
        proposals_str = ""
        if attribute_explanations is not None and not attribute_explanations.empty:
            try:
                cols = [
                    c
                    for c in ("name", "command", "argumentation", "status")
                    if c in attribute_explanations.columns
                ]
                proposals_str = attribute_explanations[cols].to_csv(index=False)
            except Exception:
                proposals_str = ""

        @wandb_logging_wrapper
        def _log() -> None:
            import wandb

            # See safe_wandb_log docstring for why step= is omitted here:
            # the LLM client's per-call token logs auto-advance the run step,
            # so passing step=index (small) makes wandb silently drop the call.
            safe_wandb_log(
                {
                    "extractor_abort/iteration_global": index,
                    "extractor_abort/iteration_in_fold": iteration_in_fold,
                    "extractor_abort/fold_run_id": fold_run_id,
                    "extractor_abort/active_features_before": num_known_attributes,
                    "extractor_abort/append_attempts_total": counters["append_total"],
                    "extractor_abort/append_attempts_success": counters[
                        "append_success"
                    ],
                    "extractor_abort/append_attempts_failed": counters["append_failed"],
                    "extractor_abort/pandas_attempts_total": counters["pandas_total"],
                    "extractor_abort/pandas_attempts_failed": counters["pandas_failed"],
                    "extractor_abort/focus_area": focus,
                    "extractor_abort/reason": reason,
                    "extractor_abort/proposed_explanations_csv": proposals_str,
                }
            )

            table = wandb.Table(
                columns=[
                    "iteration_global",
                    "iteration_in_fold",
                    "fold_run_id",
                    "reason",
                    "active_features_before",
                    "append_attempts_total",
                    "append_attempts_success",
                    "append_attempts_failed",
                    "pandas_attempts_total",
                    "pandas_attempts_failed",
                    "focus_area",
                ]
            )
            table.add_data(
                index,
                iteration_in_fold,
                fold_run_id,
                reason,
                num_known_attributes,
                counters["append_total"],
                counters["append_success"],
                counters["append_failed"],
                counters["pandas_total"],
                counters["pandas_failed"],
                focus,
            )
            safe_wandb_log({f"extractor_aborts/iteration_{index}": table})

        _log()

    def compute_fold_aggregation(self) -> dict:
        if self.df_test_pool is None or self.df_test_pool.empty:
            return {}

        completed_results = self.df_test_pool[
            self.df_test_pool["status"] == "completed"
        ].copy()
        if completed_results.empty:
            return {}

        metric_name = "accuracy" if self.cfg.task_type == "classification" else "n_RMSE"
        if metric_name not in completed_results.columns:
            return {}

        if "test_fold_id" not in completed_results.columns:
            return {}

        n_eval_folds = task_registry.evaluated_fold_count(
            rouge_one_cfg.k_folds, rouge_one_cfg.smoke_folds
        )
        expected_fold_ids = set(range(n_eval_folds))
        present_fold_ids = set(
            pd.to_numeric(
                completed_results["test_fold_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )
        if present_fold_ids != expected_fold_ids:
            return {}

        # Which per-iteration metric decides *which* iteration is reported. Under
        # "inner_cv" the decision is made on a cross-validation score computed inside
        # the training folds, so the reported metric never influences its own selection
        # (nested CV). Under "test" the legacy behaviour is kept: the held-out metric
        # both selects and is reported, which is the protocol reviewers objected to.
        selection_metric_name = (
            "inner_cv_accuracy"
            if self.cfg.task_type == "classification"
            else "inner_cv_RMSE"
        )
        use_inner_selection = (
            rouge_one_cfg.selection_signal == "inner_cv"
            and selection_metric_name in completed_results.columns
            and pd.to_numeric(completed_results[selection_metric_name], errors="coerce")
            .notna()
            .any()
        )
        # Direction is the same for either signal: accuracy-like metrics are maximised,
        # RMSE-like metrics minimised.
        minimise_selection = self.cfg.task_type != "classification"

        if rouge_one_cfg.selection_signal == "inner_cv" and not use_inner_selection:
            ConsoleManager.console_error_print(
                f"selection_signal=inner_cv requested but '{selection_metric_name}' is "
                "unavailable; falling back to the held-out metric for iteration "
                "selection. Reported numbers are NOT nested-CV clean."
            )

        active_selection_column = (
            selection_metric_name if use_inner_selection else metric_name
        )

        best_rows: list[pd.Series] = []
        for fold_id in range(n_eval_folds):
            fold_rows = completed_results[completed_results["test_fold_id"] == fold_id]
            if fold_rows.empty:
                continue

            fold_metric_values = pd.to_numeric(
                fold_rows[active_selection_column], errors="coerce"
            )

            if fold_metric_values.notna().any():
                if minimise_selection:
                    best_index = fold_metric_values.idxmin()
                else:
                    best_index = fold_metric_values.idxmax()
            else:
                # Keep aggregation robust when a fold completed but produced NaN metrics.
                best_index = fold_rows.index[-1]
            best_rows.append(completed_results.loc[best_index])

        if not best_rows:
            return {}

        if len(best_rows) != n_eval_folds:
            return {}

        best_df = (
            pd.DataFrame(best_rows).sort_values("test_fold_id").reset_index(drop=True)
        )

        def _aggregate_metric(column: str) -> dict[str, float]:
            if column not in best_df.columns:
                return {"mean": float("nan"), "std": float("nan")}
            numeric = pd.to_numeric(best_df[column], errors="coerce").dropna()
            if numeric.empty:
                return {"mean": float("nan"), "std": float("nan")}
            return {
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0)),
            }

        active_features_df = self.df_attributes_explanations[
            self.df_attributes_explanations["Status"].fillna("Active") != "Pruned"
        ].copy()
        active_features = [
            {
                "name": row.get("Attribute", ""),
                "description": row.get("Description", ""),
                "pandas_command": row.get("Pandas Command", ""),
            }
            for _, row in active_features_df.iterrows()
        ]

        aggregation = {
            "dataset_id": self.cfg.experiment_name,
            "task_type": self.cfg.task_type,
            "fold_count": len(best_df),
            "best_iterations": best_df.to_dict(orient="records"),
            "active_features": active_features,
            "feature_count": len(active_features),
        }

        if self.cfg.task_type == "classification":
            for column in ["accuracy", "precision", "recall", "f1", "AUROC"]:
                aggregation[column] = _aggregate_metric(column)
        else:
            for column in ["n_RMSE", "RMSE", "MSE", "MAE", "R2"]:
                aggregation[column] = _aggregate_metric(column)

        return aggregation

    async def report_fold_aggregation_to_wandb(self, aggregation: dict) -> None:
        if not self.enable_wandb or not aggregation:
            return

        @wandb_logging_wrapper
        def log_aggregation() -> None:
            import wandb

            summary_payload = {
                "fold_aggregation/dataset_id": aggregation.get("dataset_id"),
                "fold_aggregation/task_type": aggregation.get("task_type"),
                "fold_aggregation/fold_count": aggregation.get("fold_count"),
                "fold_aggregation/feature_count": aggregation.get("feature_count"),
            }

            metric_columns = (
                ["accuracy", "precision", "recall", "f1", "AUROC"]
                if aggregation.get("task_type") == "classification"
                else ["n_RMSE", "RMSE", "MSE", "MAE", "R2"]
            )
            for metric in metric_columns:
                metric_summary = aggregation.get(metric)
                if isinstance(metric_summary, dict):
                    summary_payload[f"fold_aggregation/{metric}_mean"] = (
                        metric_summary.get("mean")
                    )
                    summary_payload[f"fold_aggregation/{metric}_std"] = (
                        metric_summary.get("std")
                    )

            wandb.log(summary_payload)

            best_columns = ["test_fold_id", "id", *metric_columns]
            best_table = wandb.Table(columns=best_columns)
            for row in aggregation.get("best_iterations", []):
                best_table.add_data(*[row.get(column) for column in best_columns])
            wandb.log({"fold_aggregation/best_iterations": best_table})

            feature_table = wandb.Table(
                columns=["feature_name", "description", "pandas_command"]
            )
            for feature in aggregation.get("active_features", []):
                feature_table.add_data(
                    feature.get("name", ""),
                    feature.get("description", ""),
                    feature.get("pandas_command", ""),
                )
            wandb.log({"fold_aggregation/active_features": feature_table})

        log_aggregation()

    def print_fold_aggregation_to_terminal(self, aggregation: dict) -> None:
        if not aggregation:
            return

        ConsoleManager.console_rule("5-Fold Aggregation Summary")
        ConsoleManager.console_print(
            f"Dataset: {aggregation.get('dataset_id')}\n"
            f"Task Type: {aggregation.get('task_type')}\n"
            f"Completed Folds: {aggregation.get('fold_count')}"
        )

        best_iterations = aggregation.get("best_iterations", [])
        if best_iterations:
            metric_columns = (
                ["test_fold_id", "id", "accuracy", "precision", "recall", "f1", "AUROC"]
                if aggregation.get("task_type") == "classification"
                else ["test_fold_id", "id", "n_RMSE", "RMSE", "MSE", "MAE", "R2"]
            )
            best_df = pd.DataFrame(best_iterations)
            best_df = best_df[
                [column for column in metric_columns if column in best_df.columns]
            ]
            ConsoleManager.print_dataframe_as_table(
                best_df,
                title="Best Iteration Per Fold",
                style="green",
            )

        metric_rows: list[dict[str, str]] = []
        metric_columns = (
            ["accuracy", "precision", "recall", "f1", "AUROC"]
            if aggregation.get("task_type") == "classification"
            else ["n_RMSE", "RMSE", "MSE", "MAE", "R2"]
        )
        for metric in metric_columns:
            metric_summary = aggregation.get(metric)
            if not isinstance(metric_summary, dict):
                continue
            metric_rows.append(
                {
                    "metric": metric,
                    "mean": f"{metric_summary.get('mean', float('nan')):.6g}",
                    "std": f"{metric_summary.get('std', float('nan')):.6g}",
                }
            )

        if metric_rows:
            ConsoleManager.print_dataframe_as_table(
                pd.DataFrame(metric_rows),
                title="Aggregated Metrics",
                style="cyan",
            )

        feature_rows = aggregation.get("active_features", [])
        if feature_rows:
            feature_df = pd.DataFrame(feature_rows)
            ConsoleManager.print_dataframe_as_table(
                feature_df,
                title="Active Features and Definitions",
                style="magenta",
            )

    def save_fold_aggregation_csv(self, aggregation: dict) -> list[Path]:
        if not aggregation:
            return []

        output_dir = self.cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_columns = (
            ["accuracy", "precision", "recall", "f1", "AUROC"]
            if aggregation.get("task_type") == "classification"
            else ["n_RMSE", "RMSE", "MSE", "MAE", "R2"]
        )

        summary_rows = []
        for metric in metric_columns:
            metric_summary = aggregation.get(metric)
            if not isinstance(metric_summary, dict):
                continue
            summary_rows.append(
                {
                    "dataset_id": aggregation.get("dataset_id"),
                    "task_type": aggregation.get("task_type"),
                    "fold_count": aggregation.get("fold_count"),
                    "feature_count": aggregation.get("feature_count"),
                    "metric": metric,
                    "mean": metric_summary.get("mean"),
                    "std": metric_summary.get("std"),
                }
            )

        summary_path = output_dir / "fold_aggregation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        best_iterations_path = output_dir / "fold_aggregation_best_iterations.csv"
        pd.DataFrame(aggregation.get("best_iterations", [])).to_csv(
            best_iterations_path,
            index=False,
        )

        features_path = output_dir / "fold_aggregation_active_features.csv"
        pd.DataFrame(aggregation.get("active_features", [])).to_csv(
            features_path,
            index=False,
        )

        return [summary_path, best_iterations_path, features_path]

    async def log_fold_aggregation_artifacts(self, csv_paths: list[Path]) -> None:
        if not self.enable_wandb or not csv_paths:
            return

        @wandb_logging_wrapper
        def log_aggregation_artifacts() -> None:
            import wandb

            artifact = wandb.Artifact(
                name="fold_aggregation_csv",
                type="dataset",
            )
            for path in csv_paths:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)

        log_aggregation_artifacts()

    async def save_data(self):
        dst_path = self.cfg.output_dir

        df_test_pool_path = dst_path / "df_test_pool_final.csv"
        df_entities_attributes_path = dst_path / "df_entities_attributes_final.csv"
        df_attributes_explanations_path = (
            dst_path / "df_attributes_explanations_final.csv"
        )

        self.df_test_pool.to_csv(df_test_pool_path, index=False)
        self.df_entities_attributes.to_csv(df_entities_attributes_path, index=False)
        self.df_attributes_explanations.to_csv(
            df_attributes_explanations_path, index=False
        )

        if self.enable_wandb:

            @wandb_logging_wrapper
            def log_wandb_metrics():
                import wandb

                artifact = wandb.Artifact(name="df_test_pool_final", type="dataset")
                artifact.add_file(str(df_test_pool_path))
                wandb.log_artifact(artifact)

                artifact = wandb.Artifact(
                    name="df_attributes_explanations_final", type="dataset"
                )
                artifact.add_file(str(df_attributes_explanations_path))
                wandb.log_artifact(artifact)

            log_wandb_metrics()

    def _replay_validity_results(self) -> dict[str, Any]:
        """E6 replay-validity check (A03-T5 f).

        ``replay_nan_fills`` (run-total replay NaN fills in ``fold_safe_features``) is
        always reported. With ``ROGUE_ONE_REPLAY_VALIDITY_CHECK=1`` this also rescores
        each fold's selected trial from its authoring frame with ``analyse()`` of
        ``paper/scripts/diag_finding39_authoring_frame.py`` (on this run's own outer
        folds) and reports ``replay_nonrepro_folds`` = folds with ``reproduced == False``.
        Never raises.
        """
        out: dict[str, Any] = {
            "replay_nan_fills": int(instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER))
        }
        if not rouge_one_cfg.replay_validity_check:
            return out
        out["replay_check_ran"] = 0
        try:
            import importlib.util

            import numpy as np

            rel =Path("paper") / "scripts" / "diag_finding39_authoring_frame.py"
            candidates = [Path(__file__).resolve().parent / rel, Path("/workspace") / rel]
            script = next((p for p in candidates if p.is_file()), None)
            if script is None:
                raise FileNotFoundError(f"{rel} not found")
            spec = importlib.util.spec_from_file_location("_ro_diag_finding39", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            out_dir = Path(self.cfg.output_dir)
            rows = module.analyse(
                Path(self.cfg.feature_file_path).parent,
                self.cfg.task_type,
                rouge_one_cfg.k_folds,
                rouge_one_cfg.fold_seed,
                fixed_test_ids=self._tabular_fixed_test_ids,
                out_dir=out_dir,
                iterations_per_fold=rouge_one_cfg.num_iterations,
                balanced=rouge_one_cfg.do_balancing,
                scored_only=True,
                tester_row_order=True,
            )
            verdicts = [
                bool(r["reproduced"])
                for r in rows
                if isinstance(r.get("reproduced"), (bool, np.bool_))
            ]
            out["replay_nonrepro_folds"] = int(sum(1 for v in verdicts if not v))
            out["replay_checked_folds"] = len(verdicts)
            out["replay_unchecked_folds"] = len(rows) - len(verdicts)
            out["replay_check_ran"] = 1
            pd.DataFrame(rows).to_csv(out_dir / "replay_validity.csv", index=False)
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never break a run
            out["replay_check_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return out

    def _report_replay_validity(self) -> dict[str, Any]:
        results = self._replay_validity_results()
        try:
            (Path(self.cfg.output_dir) / "replay_validity_summary.json").write_text(
                json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            pass
        ConsoleManager.console_print(f"Replay validity: {results}")
        if self.enable_wandb:

            @wandb_logging_wrapper
            def log_replay_summary():
                import wandb

                for key, value in results.items():
                    wandb.run.summary[key] = value

            log_replay_summary()
        return results

    def _report_scientist_signal(self) -> dict[str, Any]:
        """E14: log ``scientist_signal`` and, under L1, the 01 4.4 item 2 split audit.

        Under ``train_fit`` only the W&B summary key is written (no file).
        """
        results = scientist_signal.audit_summary(rouge_one_cfg.scientist_signal)
        if rouge_one_cfg.scientist_signal == scientist_signal.INNER_CV_L1:
            try:
                (Path(self.cfg.output_dir) / "scientist_signal_summary.json").write_text(
                    json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception:
                pass
            ConsoleManager.console_print(f"Scientist signal: {results}")
        if self.enable_wandb:

            @wandb_logging_wrapper
            def log_signal_summary():
                import wandb

                for key, value in results.items():
                    wandb.run.summary[key] = value

            log_signal_summary()
        return results

    async def main(self):
        ConsoleManager.reset_progress_bar()
        total_trials = (
            1
            if self._tabular_skip_reason
            # E1b: k_folds * num_iterations, or smoke_folds * num_iterations under
            # ROGUE_ONE_SMOKE_FOLDS (smoke / Stage-0 runs only).
            else task_registry.total_trials(
                rouge_one_cfg.k_folds,
                rouge_one_cfg.num_iterations,
                rouge_one_cfg.smoke_folds,
            )
        )

        self._dataset_run_status = "running"
        self._dataset_failure_reason = None

        with Live(
            ConsoleManager().progress_bar,
            console=ConsoleManager().console,
            vertical_overflow="ellipsis",
        ) as live:

            ConsoleManager.progress_bar_add_task(
                description="Rogue One Agent Network Progress",
                total=total_trials,
            )

            for i in range(total_trials):
                ConsoleManager.progress_bar_update(completed=i)
                trial_status = await self.forward_pass(i)

                if trial_status == "skipped":
                    # Whole-dataset skip (e.g. fold layout impossible).
                    self._dataset_run_status = "skipped"
                    break

                if trial_status == "iteration_skipped":
                    # Mid-fold idea exhaustion — the Extractor produced 0
                    # productive features at iter > 0, so this iteration was
                    # logged-and-skipped. The fold still has trials from
                    # earlier iterations; continue to the next trial.
                    continue

                if trial_status != "completed":
                    self._dataset_run_status = "aborted"
                    aborted_fold_id = None
                    if self.df_test_pool is not None and not self.df_test_pool.empty:
                        aborted_fold_id = self.df_test_pool.iloc[-1].get(
                            "aborted_fold_id"
                        )
                    fold_suffix = (
                        f" at fold {int(aborted_fold_id)}"
                        if pd.notna(aborted_fold_id)
                        else ""
                    )
                    self._dataset_failure_reason = (
                        f"Dataset evaluation aborted{fold_suffix} during trial {i}."
                    )
                    break

            if self._dataset_run_status == "running":
                self._dataset_run_status = "completed"

            ConsoleManager.console_rule("Final Results")
            await self.save_data()
            ConsoleManager.print_dataframe_as_table(
                self.df_test_pool, title="Final Test Pool"
            )

            # Compute and report fold aggregation
            if self._dataset_run_status == "completed":
                ConsoleManager.console_rule("Computing Fold Aggregation")
                aggregation = self.compute_fold_aggregation()

                # Report to terminal
                self.print_fold_aggregation_to_terminal(aggregation)

                # Persist aggregation as CSV artifacts
                csv_paths = self.save_fold_aggregation_csv(aggregation)

                # Report to WandB
                await self.report_fold_aggregation_to_wandb(aggregation)
                await self.log_fold_aggregation_artifacts(csv_paths)
            else:
                if self._dataset_run_status == "skipped":
                    ConsoleManager.console_print(
                        "[yellow]Fold aggregation skipped because the dataset was skipped.[/yellow]"
                    )
                elif self._dataset_run_status == "aborted":
                    ConsoleManager.console_error_print(
                        "Dataset evaluation aborted before all five folds completed. "
                        "Fold aggregation will not be produced."
                    )

        # E6: replay-validity check (before finalize_run, so its CSV/JSON land in run-output).
        self._report_replay_validity()

        # E14: scientist_signal (+ the L1 split audit under inner_cv_l1) in the W&B summary.
        self._report_scientist_signal()

        # E3: summary JSON (+ W&B summary and the ``run-output`` artifact) before the run closes.
        instrumentation.finalize_run(self.cfg.output_dir, upload=self.enable_wandb)

        if self.enable_wandb:

            @wandb_logging_wrapper
            def log_final_results():
                import wandb

                log_file_artifact(
                    Path(self.cfg.output_dir) / "agent_toolcall_errors.jsonl",
                    artifact_name=f"agent_toolcall_errors_{self.cfg.experiment_name}",
                    artifact_type="agent-tool-errors",
                    metadata={
                        "experiment_name": self.cfg.experiment_name,
                        "output_dir": str(self.cfg.output_dir),
                    },
                )

                wandb.finish()

            log_final_results()

        if self._dataset_run_status == "aborted":
            raise RuntimeError(
                self._dataset_failure_reason
                or "Dataset evaluation aborted before all folds completed."
            )


if __name__ == "__main__":

    # E1b: ROGUE_ONE_TASK_REGISTRY (a YAML list of config.yml paths) replaces the
    # legacy list when set; unset = the legacy 28-config list, moved verbatim to
    # rogueone.utils.task_registry.LEGACY_TASK_PATHS. A set-but-unreadable registry
    # raises here instead of silently running the legacy sweep.
    paths = task_registry.task_paths()

    def select_paths(all_paths: list[Path]) -> list[Path]:
        """Restrict the sweep to ROGUE_ONE_DATASETS, matching on the config's parent dir."""
        return task_registry.select_paths(
            all_paths,
            rouge_one_cfg.dataset_filter,
            on_missing=lambda name: ConsoleManager.console_error_print(
                f"ROGUE_ONE_DATASETS requested '{name}' but no config matched it."
            ),
        )

    # E9c: a file-mode planner row with a bad focus file is a setup fault (A09 §4.7):
    # stop with rc != 0 before any server start or LLM call. No-op under "llm".
    if rouge_one_cfg.planner_mode in planner_mode.FILE_MODES:
        try:
            planner_mode.load_focus_plan(
                rouge_one_cfg.planner_mode,
                rouge_one_cfg.planner_focus_file,
                dataset_filter=rouge_one_cfg.dataset_filter,
                k_folds=rouge_one_cfg.k_folds,
                num_iterations=rouge_one_cfg.num_iterations,
            )
        except planner_mode.PlannerFocusError as exc:
            # SystemExit(str) prints to stderr (container log) and exits with rc 1.
            raise SystemExit(f"Planner focus file rejected: {exc}")

    async def run_experiments():
        selected = select_paths(paths)
        if not selected:
            ConsoleManager.console_error_print(
                "No experiment configs selected; nothing to run."
            )
            return
        for p in selected:
            try:
                cfg = ExperimentConfig.from_yaml(p)
                await RogueOneAgentNetwork(cfg).main()
            except Exception as e:
                ConsoleManager.console_error_print(
                    f"Error running experiment for config {p}: {type(e).__name__}: {e}",
                    # include_traceback=True,
                    # exception=e,
                )

    if env_as_bool("ROGUE_ONE_START_VLLM", True):
        start_vllm_servers(
            start_embedding_server=False,
        )
    asyncio.run(run_experiments())

    web_hook_url = os.environ.get("ROGUE_ONE_WEBHOOK_URL", "")
    message = "RUN FINISHED"

    if web_hook_url:
        try:
            requests.post(web_hook_url, json={"text": message})
        except Exception as e:
            ConsoleManager.console_error_print(
                f"Error sending webhook notification: {type(e).__name__}: {e}"
            )
