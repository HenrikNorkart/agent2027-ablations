"""Fold-safe feature recomputation -- closes the architectural half of Finding 14
(2026-08-25, `paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md`).

`RogueOneOrchestrator` computes every engineered feature once, via `self.df` (the
full, un-split dataset spanning every outer fold), and only partitions rows into
folds *afterward*, purely for scoring. That is harmless for a row-wise feature
(e.g. ``df['a'] / df['b']``): its value for any row never depends on any other
row, so computing it globally or per-partition gives identical numbers. It is
NOT harmless for a feature whose command aggregates across rows
(``groupby(...).transform(...)``, ``.rank()``, ``.rolling()``, frequency/count
encodings, ...): the statistic it computes is fit using every row `self.df` has,
including whichever rows happen to be *this* outer fold's held-out test set --
so a fold's own held-out rows can (and, per Finding 14, did) contribute to a
feature that then gets used to train on and score against that very fold.

The fix here is not a fit/transform rewrite of the Extractor's tool -- that
would require the LLM-authored code to declare which parts are "fit on train"
vs "apply to any row", which the current `append_new_attribute` grammar does
not support. Instead, every attribute's stored Pandas Command is *replayed*
against one partition at a time: once against a fold's train rows alone, once
against its test rows alone (see `build_fold_safe_entities_folds`). A row-wise
command produces byte-identical values either way. An aggregating command now
fits its statistic using only the rows in the partition it is applied to, so a
fold's test rows never inform (or benefit from) an aggregate that also feeds
that same fold's training data.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rogueone.utils import instrumentation
from rogueone.utils.console import ConsoleManager
from rogueone.utils.pandas_exec import exec_user_command as _exec_user_command

#: E6: run-total counter of replay NaN fills (the three sites below). Read through
#: ``instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER)``; reset per run by
#: ``instrumentation.configure``.
REPLAY_NAN_FILLS_COUNTER = "replay_nan_fills"


def materialize_fold_safe_features(
    df_raw: pd.DataFrame,
    attribute_explanations: pd.DataFrame,
    row_mask: np.ndarray,
) -> pd.DataFrame:
    """Replay every recorded attribute's command against `df_raw[row_mask]` alone.

    Args:
        df_raw: The full raw dataframe (`self.df` -- id + raw columns only, no
            engineered columns, no target).
        attribute_explanations: `self.df_attributes_explanations` -- one row per
            attribute ever added, with `Attribute` and `Pandas Command` columns.
            Row order must match creation order (true for
            `RogueOneOrchestrator.update_attribute_pool`, which only appends).
        row_mask: Boolean mask over `df_raw`'s rows selecting the partition to
            compute against -- e.g. a fold's train rows, or its test rows.
            Never a mix of both, or the leak this module exists to close comes
            right back.

    Returns:
        A DataFrame with `len(df_raw[row_mask])` rows (index reset to 0..n-1,
        matching `df_raw[row_mask]`'s row order), an `id` column, and one column
        per attribute in `attribute_explanations`. A column that fails to
        recompute is filled with NaN rather than aborting the run.

    Finding 38 (2026-09-15): the replay frame must carry `id`. At authoring time
    the Extractor's `df_features` IS `self.df_entities_attributes`, which is
    seeded with the entity-id column before any feature is added, and the
    Extractor's own tool docs push the model toward
    ``result = df_features['id'].map(df.groupby('id')['age'].first())`` as the
    canonical shape (`extractor/tools.py`: a command "must include a
    groupby('id') operation so the result aligns with df_features"). Replaying
    that command against a frame with no `id` column raises KeyError, which this
    function NaN-fills -- so the fold-safe path was quietly discarding the
    majority of every dataset's engineered features. Measured over the 27
    seed-42 run directories of the 2026-09-11 wave: 3397 attribute replays
    failed without the `id` column, 30 with it (2453 of the failures were
    literally `KeyError: 'id'`, the rest cascades off them).
    """
    sub_df = df_raw[row_mask].reset_index(drop=True)
    sub_features = pd.DataFrame(index=sub_df.index)
    if "id" in sub_df.columns:
        sub_features["id"] = sub_df["id"].to_numpy()

    for _, row in attribute_explanations.iterrows():
        name = row["Attribute"]
        command = row.get("Pandas Command", "")
        if not isinstance(command, str) or not command.strip():
            ConsoleManager.console_error_print(
                f"[fold_safe_features] attribute '{name}' has no recorded Pandas "
                "Command; filling NaN for this partition rather than aborting."
            )
            instrumentation.count(REPLAY_NAN_FILLS_COUNTER)
            sub_features[name] = np.nan
            continue
        try:
            result, _ = _exec_user_command(command, sub_df, sub_features)
        except Exception as exc:  # noqa: BLE001 - a bad replay must not crash a trial
            ConsoleManager.console_error_print(
                f"[fold_safe_features] failed to recompute '{name}' on this "
                f"partition ({len(sub_df)} rows): {exc!r}. Filling NaN."
            )
            instrumentation.count(REPLAY_NAN_FILLS_COUNTER)
            sub_features[name] = np.nan
            continue

        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.reset_index(drop=True).values

        # The assignment itself has to be guarded too, not just the exec above.
        # A command that mixes an id-indexed object with a positionally-indexed
        # one (``df.groupby('id')['pH'].first() / df_features['total_acidity']``)
        # makes pandas align on the UNION of the two indexes, so `result` comes
        # back longer than the partition and this assignment raises
        # `ValueError: Length of values (N) does not match length of index (M)`.
        # Before Finding 38 that escaped the try/except, propagated out of
        # `build_fold_safe_entities_folds`, and was caught by the orchestrator's
        # `except ValueError` as a whole-dataset SKIP -- which is how bike,
        # covtype, heart, housing, insurance and wine produced no
        # fold_aggregation_summary.csv in the 2026-09-11 wave despite rc=0.
        # NaN-filling one attribute is the documented fallback; losing the
        # dataset is not.
        try:
            sub_features[name] = result
        except Exception as exc:  # noqa: BLE001 - a bad replay must not kill a run
            ConsoleManager.console_error_print(
                f"[fold_safe_features] could not store '{name}' on this "
                f"partition ({len(sub_df)} rows): {exc!r}. Filling NaN."
            )
            instrumentation.count(REPLAY_NAN_FILLS_COUNTER)
            sub_features[name] = np.nan

    return sub_features


def build_fold_safe_entities_folds(
    df_raw: pd.DataFrame,
    attribute_explanations: pd.DataFrame,
    labels_df: pd.DataFrame,
    fixed_test_ids: Sequence[set],
    current_test_fold_id: int,
    id_col: str = "id",
) -> list[pd.DataFrame]:
    """Build the per-fold dataframes for ONE `test_hypotheses` call, fold-safely.

    Must be called once per distinct `current_test_fold_id` (i.e. once per
    outer fold-run), not once for the whole dataset: a given fold plays "train"
    in `k_folds - 1` calls and "test" in exactly 1, and only the call where it
    is "test" needs its rows computed from their own partition alone. The other
    `k_folds - 1` folds are computed together as one train partition (excluding
    `current_test_fold_id`'s rows), matching how `TesterAgent.test_hypotheses`
    subsequently vstacks every non-selected fold into one training set anyway.

    Args:
        df_raw: The full raw dataframe (`self.df`).
        attribute_explanations: `self.df_attributes_explanations`.
        labels_df: id -> target lookup (`self.splitter.labels`, already
            column-normalized to `id_col` by the caller).
        fixed_test_ids: One id-set per outer fold, in fold order (`0..k-1`).
        current_test_fold_id: Which fold in `fixed_test_ids` is "test" for this
            call -- matches the `step` / `fold_run_id` passed to
            `test_hypotheses`.
        id_col: Name of the id column in `df_raw` and `labels_df`.

    Returns:
        A list of DataFrames, one per fold in `fixed_test_ids` order, each with
        `id_col`, `target`, and one column per attribute -- structurally
        identical to the plain `self.df_entities_attributes`-slicing this
        replaces, just leak-free.
    """
    current_test_ids = fixed_test_ids[current_test_fold_id]
    is_current_test = df_raw[id_col].isin(current_test_ids).to_numpy()

    train_features = materialize_fold_safe_features(
        df_raw, attribute_explanations, ~is_current_test
    )
    train_features[id_col] = df_raw.loc[~is_current_test, id_col].reset_index(
        drop=True
    )

    test_features = materialize_fold_safe_features(
        df_raw, attribute_explanations, is_current_test
    )
    test_features[id_col] = df_raw.loc[is_current_test, id_col].reset_index(drop=True)

    labels_unique = labels_df.drop_duplicates(subset=[id_col])

    folds: list[pd.DataFrame] = []
    for fold_id, test_ids in enumerate(fixed_test_ids):
        source = test_features if fold_id == current_test_fold_id else train_features
        fold_df = source[source[id_col].isin(test_ids)].reset_index(drop=True)
        fold_df = fold_df.merge(labels_unique, on=id_col, how="left")
        folds.append(fold_df)
    return folds


# --------------------------------------------------------------------------- E6 augment seeding
# AGENT 2027 E6 / A03-T5: ``ROGUE_ONE_FEATURE_POOL_MODE=augment`` seeds every raw column
# of ``data.csv`` (except ``id``) as a replayable, prune-protected attribute row, so the
# iteration-0 pool equals the input of the ``base_no_fe`` baseline
# (``paper/scripts/classicfe_arm.py``: ``prep()`` then ``to_matrix`` / ``nan_to_num``).
#
# Deliberately NOT built on the Extractor's baseline fallback or its
# ``_ordinal_encode_categorical`` (A03 §2): those cap at 8 features, skip columns with
# NaN, add interactions, and re-derive the category map from whichever partition the
# command is replayed on. Here the category -> code map is a literal embedded in the
# command, built once over the full column, so a replay on any partition is exact.

#: ``Added`` marker of a seeded attribute row.
SEED_ADDED_MARKER = "Seed"
#: Boolean attribute-table column flagging seeded rows. Only added under augment, so the
#: replace pool stays byte-identical to pre-E6; a missing column means "no seeds".
SEEDED_COLUMN = "Seeded"
#: Allowed values of ``ROGUE_ONE_FEATURE_POOL_MODE`` (first = code default).
FEATURE_POOL_MODES = ("replace", "augment")


def is_categorical_raw_column(series: pd.Series) -> bool:
    """A raw column is categorical iff it is not numeric.

    Not ``dtype == object``: pandas 3 (the Rogue One image) reads string columns as the
    ``str`` dtype, so an ``object`` test finds no categoricals there, while the
    ``base_no_fe`` reference (pandas 2.2.3) sees them as ``object`` (A03 §2 dtype trap).
    """
    return not pd.api.types.is_numeric_dtype(series)


def seed_command_for_raw_column(
    df_raw: pd.DataFrame, column: str
) -> tuple[str, str, str]:
    """``(attribute_name, pandas_command, description)`` for one raw column.

    Numeric: the canonical id-aligned shape (Finding 38), missing -> 0.
    Categorical: a literal alphabetical category -> code map built over the full column
    (as ``classicfe_arm.prep``), named ``<col>_ord``, missing -> 0.
    Both end in ``.astype(float)``, so a bool raw column cannot turn the Tester's
    feature matrix into an ``object`` array.
    """
    col_lit = repr(str(column))
    series = df_raw[column]
    if is_categorical_raw_column(series):
        categories = sorted(series.dropna().astype(str).unique().tolist())
        mapping = {c: i for i, c in enumerate(categories)}
        command = (
            f"result = df_features['id'].map(df.groupby('id')[{col_lit}].first()"
            f".astype(str).map({mapping!r})).fillna(0.0).astype(float)"
        )
        description = (
            f"Raw categorical input column {col_lit}, ordinal-encoded by alphabetical "
            "category order over the full column (missing -> 0)."
        )
        return f"{column}_ord", command, description
    command = (
        f"result = df_features['id'].map(df.groupby('id')[{col_lit}].first())"
        ".fillna(0.0).astype(float)"
    )
    description = f"Raw input column {col_lit}, unchanged (missing -> 0)."
    return str(column), command, description


def seed_raw_feature_rows(
    df_raw: pd.DataFrame,
    df_entities: pd.DataFrame,
    df_attributes: pd.DataFrame,
    *,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E6 augment seeding: the one pure function ``_reset_fold_state`` and
    ``scripts/agent2027/e6_seed_check.py`` both call.

    Adds one column per raw column of ``df_raw`` except ``id_col`` to a copy of
    ``df_entities`` (values computed by executing the seed's own command, so the
    authoring frame and every replay run the same code), and one matching row to a
    copy of ``df_attributes`` with ``Status="Active"``, ``Added="Seed"`` and
    ``Seeded=True`` (all other rows get ``Seeded=False``). Rows are never reordered and
    the inputs are not mutated. Names already present in ``df_entities`` are skipped.
    """
    entities = df_entities.copy()
    attributes = df_attributes.copy()
    if SEEDED_COLUMN not in attributes.columns:
        attributes[SEEDED_COLUMN] = False
    if id_col not in df_raw.columns or id_col not in entities.columns:
        ConsoleManager.console_error_print(
            "[E6 seeding] no id column in the raw or entity frame; nothing seeded."
        )
        return entities, attributes

    order_before = entities[id_col].tolist()
    new_rows: list[dict] = []
    new_columns: dict[str, np.ndarray] = {}
    for column in df_raw.columns:
        if column == id_col:
            continue
        name, command, description = seed_command_for_raw_column(df_raw, column)
        if name in entities.columns or name in new_columns:
            continue
        result, _ = _exec_user_command(command, df_raw, entities)
        values = pd.Series(result).reset_index(drop=True).to_numpy(dtype=float)
        if len(values) != len(entities) or np.isnan(values).any():
            raise ValueError(
                f"E6 seed command for raw column {column!r} did not align with the "
                "entity frame"
            )
        new_columns[name] = values
        new_rows.append(
            {
                "Attribute": name,
                "Description": description,
                "Pandas Command": command,
                "Status": "Active",
                "Added": SEED_ADDED_MARKER,
                SEEDED_COLUMN: True,
            }
        )

    if new_columns:
        entities = pd.concat(
            [entities, pd.DataFrame(new_columns, index=entities.index)], axis=1
        )
    if entities[id_col].tolist() != order_before:  # pragma: no cover - defensive
        raise RuntimeError("E6 seeding reordered the entity rows")

    if new_rows:
        seeds = pd.DataFrame(new_rows).reindex(columns=attributes.columns)
        # Columns the seeds leave empty (e.g. "Pruned Iteration") would be float64 NaN;
        # under pandas 3 a later string write into them raises, so keep them object.
        for col in seeds.columns:
            if col != SEEDED_COLUMN and seeds[col].isna().all():
                seeds[col] = seeds[col].astype(object)
        attributes = (
            pd.concat([seeds, attributes], ignore_index=True)
            if len(attributes)
            else seeds
        )
    attributes[SEEDED_COLUMN] = (
        attributes[SEEDED_COLUMN].astype("boolean").fillna(False).astype(bool)
    )
    return entities, attributes.reset_index(drop=True)


def seeded_attribute_mask(df_attributes: pd.DataFrame) -> pd.Series:
    """Boolean mask (aligned to ``df_attributes``) of seeded rows; all False without seeds."""
    if SEEDED_COLUMN not in df_attributes.columns:
        return pd.Series(False, index=df_attributes.index, dtype=bool)
    return df_attributes[SEEDED_COLUMN].astype("boolean").fillna(False).astype(bool)


def seeded_attribute_names(df_attributes: pd.DataFrame) -> set[str]:
    """Names of the seeded rows (empty set without seeds)."""
    mask = seeded_attribute_mask(df_attributes)
    return set(df_attributes.loc[mask, "Attribute"].astype(str))


def raw_coverage(
    df_raw: pd.DataFrame, df_attributes: pd.DataFrame, *, id_col: str = "id"
) -> float:
    """Share of raw columns whose seeded row is ``Active``; 0.0 under replace."""
    n_raw = sum(1 for c in df_raw.columns if c != id_col)
    if n_raw == 0 or "Status" not in df_attributes.columns:
        return 0.0
    mask = seeded_attribute_mask(df_attributes)
    active = df_attributes["Status"].astype(str).str.strip().str.lower() != "pruned"
    return float(int((mask & active).sum())) / float(n_raw)
