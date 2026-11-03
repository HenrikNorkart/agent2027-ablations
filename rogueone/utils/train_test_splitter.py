from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig


def assert_disjoint_indices(
    train_indices: Iterable[int], test_indices: Iterable[int]
) -> None:
    """Validate that train and test index sets have no overlap.
    
    Args:
        train_indices: Indices for training set.
        test_indices: Indices for test set.
        
    Raises:
        ValueError: If train and test sets share any indices.
    """

    overlap = set(train_indices).intersection(set(test_indices))
    if overlap:
        raise ValueError(
            f"Train/test overlap detected for {len(overlap)} samples: {sorted(list(overlap))[:10]}"
        )



# Sentinel standing in for a missing value when a row is turned into a comparison key.
# It must be a string no real value can collide with, and it must be the SAME for every
# NaN so that two rows missing the same field still compare equal -- which is the
# behaviour duplicate detection wants: two identical rows with the same hole in them are
# duplicates, not distinct.
_NA_KEY = "\x00NA"


def _row_keys(df: pd.DataFrame, compare_cols: list[str]) -> pd.Series:
    """Join each row's values into a single comparable string key.

    Why this is not just ``df[cols].astype(str).agg("|".join, axis=1)``: under pandas 3's
    default string dtype, ``astype(str)`` leaves a missing value as a float ``nan``
    rather than converting it to ``"nan"``, so the join raises
    ``TypeError: sequence item N: expected str instance, float found`` on any frame with
    a hole in it. That failure is silent in practice -- the orchestrator catches it,
    reports "Error running experiment", and exits **rc=0** with no result -- which is the
    same silent-success mode as Findings 16 and 17. It hit 4 of the 27 reported datasets
    (arrhythmia, breast-w, eucalyptus, housing) the first time this code ran end-to-end,
    on 2026-09-11.

    Going through ``object`` dtype sidesteps the string-dtype behaviour entirely, and the
    NaNs are replaced with an explicit sentinel first so the result is deliberate rather
    than dependent on how a given pandas version renders a missing value.
    """

    sub = df[compare_cols].astype(object)
    sub = sub.where(sub.notna(), _NA_KEY)
    return sub.astype(str).agg("|".join, axis=1)


def compute_duplicate_groups(
    df: pd.DataFrame, exclude_cols: Iterable[str] = ("id",)
) -> np.ndarray:
    """Assign each row an integer group id so exact-duplicate rows share a group.

    Two rows are duplicates if every column other than `exclude_cols` (e.g. an
    opaque `id`) matches exactly. Rows with no duplicate get their own singleton
    group. Used so a K-fold split can keep duplicate rows together on one side
    of the train/test boundary -- see Finding 14 (2026-08-25): `cmc` and
    `diamonds` both have real exact-duplicate rows (14.3% / 1.3% of rows), and a
    plain shuffled KFold has no reason to keep a duplicate pair on the same
    side, so a tree-based evaluator can score a "held-out" row for free by
    memorizing its twin in the training fold.

    Args:
        df: The dataframe to group (raw feature columns; must NOT include the
            target -- two rows with identical features but different labels
            are not leakage and should not be forced together).
        exclude_cols: Columns to ignore when comparing rows for duplication.

    Returns:
        np.ndarray of shape (len(df),): group id per row, aligned to df's order.
    """
    compare_cols = [c for c in df.columns if c not in set(exclude_cols)]
    if not compare_cols:
        # Every row would key to the empty string and collapse into ONE group, which
        # `deterministic_kfold_indices` then rejects with "only 1 distinct duplicate
        # groups", skipping the whole dataset. That is not hypothetical: the original
        # call site passed `df_entities_attributes`, which holds nothing but `id` at
        # orchestrator-construction time, and it silently skipped every dataset it was
        # applied to (2026-08-28, Finding 17). Fail loudly instead -- a caller with no
        # comparable columns has a bug, and a group id it cannot act on is worse than
        # an exception.
        raise ValueError(
            "compute_duplicate_groups needs at least one column to compare, but all "
            f"of {list(df.columns)} are excluded. Pass the RAW data frame, not the "
            "engineered-feature frame (which is id-only before features are added)."
        )
    keys = _row_keys(df, compare_cols)
    group_ids, _ = pd.factorize(keys)
    return group_ids


def compute_entity_duplicate_groups(
    df: pd.DataFrame,
    entity_ids: Sequence,
    exclude_cols: Iterable[str] = ("id",),
) -> np.ndarray:
    """Duplicate-group ids for ENTITIES, aligned to `entity_ids`.

    `compute_duplicate_groups` groups rows of whatever frame it is handed. Folds,
    though, are defined over *entities* (one position per `id` in
    `df_entities_attributes`), while the raw table `df` may carry several rows per
    entity. This bridges the two: it builds a content key per raw row, folds the keys
    belonging to one entity together, and returns one group id per entry of
    `entity_ids`, in that order.

    Two entities land in the same group exactly when their raw rows are identical as a
    multiset -- which is the property Finding 14 cares about, since that is when a
    tree-based evaluator can score a held-out entity by memorizing its twin.

    Args:
        df: The RAW input table (must contain the id column named in `exclude_cols`
            unless it is indexed positionally). Must NOT include the target.
        entity_ids: The entity ids, in the exact order the feature frame lists them.
        exclude_cols: Columns to ignore when comparing rows (the id itself).

    Returns:
        np.ndarray of shape (len(entity_ids),): group id per entity.
    """
    exclude = set(exclude_cols)
    compare_cols = [c for c in df.columns if c not in exclude]
    if not compare_cols:
        raise ValueError(
            "compute_entity_duplicate_groups needs at least one column to compare, "
            f"but all of {list(df.columns)} are excluded."
        )

    row_keys = _row_keys(df, compare_cols)

    if "id" in df.columns:
        # Sort each entity's row keys so row ORDER within an entity cannot make two
        # otherwise-identical entities look different.
        per_entity = (
            pd.DataFrame({"id": df["id"], "_k": row_keys})
            .groupby("id")["_k"]
            .apply(lambda s: "\x1e".join(sorted(s)))
        )
        keys = pd.Series(list(entity_ids)).map(per_entity)
        # An entity with no raw rows cannot be compared to anything; give it a unique
        # key so it stays a singleton rather than silently joining the NaN group.
        missing = keys.isna()
        if missing.any():
            keys = keys.where(~missing, pd.Series([f"\x00missing-{i}" for i in range(len(keys))]))
    else:
        # No id column: the raw table is already one row per entity, positionally
        # aligned with the feature frame.
        if len(row_keys) != len(entity_ids):
            raise ValueError(
                f"df has {len(row_keys)} rows but there are {len(entity_ids)} "
                "entities, and df has no `id` column to join on."
            )
        keys = row_keys.reset_index(drop=True)

    group_ids, _ = pd.factorize(keys)
    return group_ids


def assert_no_group_split(
    train_indices: Iterable[int], test_indices: Iterable[int], groups: np.ndarray
) -> None:
    """Validate that no duplicate-row group straddles the train/test boundary.

    Args:
        train_indices: Row positions assigned to train.
        test_indices: Row positions assigned to test.
        groups: Per-row group id from `compute_duplicate_groups`, aligned by position.

    Raises:
        ValueError: If any group id appears on both sides.
    """
    train_groups = set(groups[list(train_indices)])
    test_groups = set(groups[list(test_indices)])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise ValueError(
            f"{len(overlap)} duplicate-row group(s) split across train/test: "
            f"{sorted(overlap)[:10]}"
        )


def deterministic_kfold_indices(
    n_rows: int,
    n_splits: int,
    random_state: int,
    stratify_labels: Sequence | None = None,
    dedup_groups: Sequence | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Generate deterministic K-fold train/test index splits.

    Uses scikit-learn's KFold with shuffle=True and specified random_state
    for reproducible splits. Validates that splits are disjoint.

    When `stratify_labels` is supplied, StratifiedKFold is used instead. This exists so
    the fold partition can be made identical to a baseline that stratifies (LLM-FE uses
    StratifiedKFold for classification), which removes the fold definition as a
    confound when comparing systems. Falls back to plain KFold, with a warning, when
    stratification is not viable -- i.e. some class has fewer members than n_splits.

    When `dedup_groups` is supplied AND it actually contains duplicate group ids
    (i.e. some row shares its group with another row), a group-aware splitter
    (GroupKFold / StratifiedGroupKFold) is used instead so that no duplicate-row
    group is ever split across a fold's train/test boundary (see Finding 14,
    2026-08-25). When every row is its own singleton group, this parameter has
    no effect and the ordinary KFold/StratifiedKFold path runs unchanged -- so a
    dataset with no duplicates gets byte-identical folds to before this fix.

    Args:
        n_rows (int): Total number of samples to fold.
        n_splits (int): Number of folds (k).
        random_state (int): Random seed for determinism.
        stratify_labels (Sequence | None): Per-row class labels, ordered to match the
            feature frame. Length must equal n_rows. None means plain KFold.
        dedup_groups (Sequence | None): Per-row duplicate-group id (see
            `compute_duplicate_groups`), ordered to match the feature frame. None or
            all-singleton means no grouping is applied.

    Returns:
        list[tuple[list[int], list[int]]]: List of (train_indices, test_indices) pairs.

    Raises:
        ValueError: If n_rows < n_splits (insufficient samples for requested folds),
            or if stratify_labels/dedup_groups is supplied with a length other than
            n_rows.
    """

    if n_rows < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds for dataset with {n_rows} rows."
        )

    y = None
    if stratify_labels is not None:
        y = np.asarray(stratify_labels)
        if y.shape[0] != n_rows:
            raise ValueError(
                f"stratify_labels has {y.shape[0]} entries but n_rows is {n_rows}; "
                "the label order must match the feature frame."
            )
        _, counts = np.unique(y, return_counts=True)
        if counts.min() < n_splits:
            ConsoleManager.console_error_print(
                f"Stratified folds requested but the rarest class has {counts.min()} "
                f"members (< n_splits={n_splits}); falling back to unstratified KFold. "
                "Fold partition will NOT match a stratifying baseline."
            )
            y = None

    groups = None
    if dedup_groups is not None:
        groups = np.asarray(dedup_groups)
        if groups.shape[0] != n_rows:
            raise ValueError(
                f"dedup_groups has {groups.shape[0]} entries but n_rows is {n_rows}; "
                "the group order must match the feature frame."
            )
        if len(np.unique(groups)) == n_rows:
            # No actual duplicates -- every row is its own group, so grouping would
            # not change anything. Skip it and take the ordinary path below.
            groups = None

    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < n_splits:
            raise ValueError(
                f"Cannot create {n_splits} folds: only {n_groups} distinct duplicate "
                f"groups for {n_rows} rows."
            )
        if y is not None:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(np.zeros((n_rows, 1)), y, groups=groups)
        else:
            splitter = GroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(np.zeros((n_rows, 1)), groups=groups)
    elif y is not None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = splitter.split(np.zeros((n_rows, 1)), y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(range(n_rows))

    fold_indices: list[tuple[list[int], list[int]]] = []
    for train_idx, test_idx in split_iter:
        train_list = train_idx.tolist()
        test_list = test_idx.tolist()
        assert_disjoint_indices(train_list, test_list)
        if groups is not None:
            assert_no_group_split(train_list, test_list, groups)
        fold_indices.append((train_list, test_list))
    return fold_indices


class TrainTestSplitter:
    """Utility for splitting data into train/test or K-fold splits.
    
    Handles label merging, deterministic shuffling, and fold generation.
    Validates train/test disjointness.
    
    Attributes:
        cfg (ExperimentConfig): Experiment configuration.
        test_size (float): Fraction of data for test set.
        random_state (int): Random seed for reproducibility.
        labels (pd.DataFrame): DataFrame containing labels for each sample.
    """
    def __init__(
        self,
        cfg: ExperimentConfig,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the TrainTestSplitter.
        
        Args:
            cfg (ExperimentConfig): Experiment configuration with label file path.
            test_size (float, optional): Fraction of data to use for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.cfg = cfg
        self.test_size = test_size
        self.random_state = random_state

        labels_file_path = cfg.label_file_path

        assert labels_file_path.exists(), f"File {labels_file_path} does not exist."

        self.labels = pd.read_csv(labels_file_path)

    async def test_train_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame into train and test sets based on configured ratio.
        
        Merges labels into the dataframe, respects existing train/test flags if present,
        otherwise splits deterministically. Prints target distribution summary.
        
        Args:
            df (pd.DataFrame): The input DataFrame to split.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames with is_train/is_test flags.
        """

        local_df = df.copy()

        # Normalize self.labels to a DataFrame with column "id"
        if isinstance(self.labels, pd.DataFrame):
            tm_target = self.labels.copy()
            if "id" not in tm_target.columns and "patient_id" in tm_target.columns:
                tm_target = tm_target.rename(columns={"patient_id": "id"})
        else:
            tm_target = pd.DataFrame(self.labels, columns=["id", "target"])

        # Merge labels into local_df on the "id" column (or on the first column if "id" is absent)
        if "id" in local_df.columns:
            local_df = local_df.merge(
                tm_target.drop_duplicates(subset=["id"]),
                on="id",
                how="left",
            )
        else:
            id_col = local_df.columns[0]
            local_df = local_df.merge(
                tm_target.drop_duplicates(subset=["id"]).rename(columns={"id": id_col}),
                on=id_col,
                how="left",
            )

        if "is_train" in local_df.columns and "is_test" in local_df.columns:
            df_train = local_df[local_df["is_train"] == True].reset_index(drop=True)
            df_test = local_df[local_df["is_test"] == True].reset_index(drop=True)

        else:
            # Shuffle deterministically
            local_df = local_df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            sep_index = int(len(local_df) * (1 - self.test_size))

            df_train = local_df.iloc[:sep_index].reset_index(drop=True)
            df_test = local_df.iloc[sep_index:].reset_index(drop=True)

            df_train["is_train"] = True
            df_train["is_test"] = False

            df_test["is_test"] = True
            df_test["is_train"] = False

        test_group_counts = df_test.groupby("target").size().reset_index(name="count")
        train_group_counts = df_train.groupby("target").size().reset_index(name="count")

        merged_counts = pd.merge(
            train_group_counts.rename(columns={"count": "train_count"}),
            test_group_counts.rename(columns={"count": "test_count"}),
            on="target",
            how="outer",
        ).fillna(0)

        merged_counts["train_count"] = merged_counts["train_count"].astype(int)
        merged_counts["test_count"] = merged_counts["test_count"].astype(int)
        merged_counts["total_count"] = (
            merged_counts["train_count"] + merged_counts["test_count"]
        )

        ConsoleManager.print_dataframe_as_table(
            merged_counts, title="Target counts (train/test)", style="cyan"
        )

        return df_train, df_test

    async def k_fold_split(self, df: pd.DataFrame, k: int) -> list[pd.DataFrame]:
        """Perform K-fold cross-validation split using deterministic KFold.
        
        Generates k folds with labels merged in, validates disjointness, and prints
        target distribution per fold.
        
        Args:
            df (pd.DataFrame): The input DataFrame to split.
            k (int): The number of folds.
        
        Returns:
            list[pd.DataFrame]: List of k DataFrames, one for each fold.
        """

        local_df = df.copy()

        # Normalize labels to a DataFrame with a column named "id"
        if isinstance(self.labels, pd.DataFrame):
            labels_df = self.labels.copy()
            if "id" not in labels_df.columns and "patient_id" in labels_df.columns:
                labels_df = labels_df.rename(columns={"patient_id": "id"})
        else:
            labels_df = pd.DataFrame(self.labels, columns=["id", "target"])

        # Merge labels into the working DataFrame
        if "id" in local_df.columns:
            labels_to_merge = labels_df.drop_duplicates(subset=["id"])
            merged_on = "id"
        else:
            merged_on = local_df.columns[0]
            labels_to_merge = labels_df.drop_duplicates(subset=["id"]).rename(
                columns={"id": merged_on}
            )

        local_df = local_df.merge(labels_to_merge, on=merged_on, how="left")

        # Exclude id/target from duplicate detection: two rows with identical raw
        # features but different labels are not leakage, and merged-on-target
        # should never itself decide the grouping.
        dedup_groups = compute_duplicate_groups(
            local_df, exclude_cols=(merged_on, "target")
        )

        fold_indices = deterministic_kfold_indices(
            n_rows=len(local_df),
            n_splits=k,
            random_state=self.random_state,
            dedup_groups=dedup_groups,
        )
        folds = [local_df.iloc[test_idx].reset_index(drop=True) for _, test_idx in fold_indices]

        # Print distribution of targets per fold

        stats = {i: {} for i in range(k)}

        match self.cfg.task_type:
            case "classification":
                for i, fold in enumerate(folds):
                    if not fold.empty and "target" in fold.columns:
                        vc = fold.groupby("target").size().reset_index(name="count")
                        stats[i]["Fold"] = f"Fold {i+1}"
                        for _, row in vc.iterrows():
                            stats[i][row["target"]] = row["count"]

            case "regression":
                for i, fold in enumerate(folds):
                    if not fold.empty and "target" in fold.columns:
                        stats[i]["Fold"] = f"Fold {i+1}"
                        stats[i]["Mean"] = f'{fold["target"].mean():.2f}'
                        stats[i]["StdDev"] = f'{fold["target"].std():.2f}'
                        stats[i]["Min"] = f'{fold["target"].min():.2f}'
                        stats[i]["Max"] = f'{fold["target"].max():.2f}'
            case _:
                raise ValueError(f"Unsupported task type: {self.cfg.task_type}")

        if not stats:
            return []

        try:
            stats_df = pd.DataFrame.from_dict(stats, orient="index").fillna(0)

            ConsoleManager.print_dataframe_as_table(
                stats_df, title="Target distribution per fold", style="cyan"
            )
        except Exception as e:
            ConsoleManager.console_print(
                f"[red]Error displaying fold statistics: {e}[/red]"
            )
        return folds
