"""Unit tests for fold-safe feature recomputation (Finding 14, 2026-08-25)."""

import numpy as np
import pandas as pd
import pytest

from rogueone.utils.fold_safe_features import (
    build_fold_safe_entities_folds,
    materialize_fold_safe_features,
)


@pytest.mark.unit
@pytest.mark.config
def test_row_wise_feature_is_identical_whether_computed_globally_or_fold_safely():
    """A row-wise command (no cross-row aggregation) must give the same values
    whether it is recomputed against a partition alone or against everything --
    this is the regression guarantee that the fix must not change ordinary
    (non-leaky) features."""
    df_raw = pd.DataFrame({"id": range(10), "a": range(10), "b": range(10, 20)})
    explanations = pd.DataFrame(
        [{"Attribute": "sum_ab", "Pandas Command": "result = df['a'] + df['b']"}]
    )

    full = materialize_fold_safe_features(
        df_raw, explanations, row_mask=np.ones(10, dtype=bool)
    )
    half_mask = np.array([True, False] * 5)
    half = materialize_fold_safe_features(df_raw, explanations, row_mask=half_mask)

    expected = df_raw.loc[half_mask, "a"] + df_raw.loc[half_mask, "b"]
    assert list(half["sum_ab"]) == list(expected)
    # And it must match the corresponding rows of the "everyone" computation too.
    assert list(half["sum_ab"]) == [
        v for v, m in zip(full["sum_ab"], half_mask) if m
    ]


@pytest.mark.unit
@pytest.mark.config
def test_aggregate_feature_does_not_leak_test_rows_into_train_partition():
    """This is the actual Finding-14 mechanism: a groupby/transform feature's
    statistic must be fit using ONLY the partition it is computed against, so a
    fold's held-out test rows never influence a value used to train on that
    same fold.
    """
    # Two groups ('x', 'y'). If group means were computed over ALL rows, both
    # partitions would see the SAME mean per group. Fold-safe computation must
    # instead give each partition its OWN mean, computed from its own rows only.
    df_raw = pd.DataFrame(
        {
            "id": range(6),
            "grp": ["x", "x", "x", "y", "y", "y"],
            "val": [10.0, 10.0, 100.0, 1.0, 1.0, 50.0],
        }
    )
    explanations = pd.DataFrame(
        [
            {
                "Attribute": "grp_mean",
                "Pandas Command": "result = df.groupby('grp')['val'].transform('mean')",
            }
        ]
    )

    train_mask = np.array([True, True, False, True, True, False])
    test_mask = ~train_mask

    train_feats = materialize_fold_safe_features(df_raw, explanations, train_mask)
    test_feats = materialize_fold_safe_features(df_raw, explanations, test_mask)

    # Train partition (rows 0,1,3,4): group x mean = 10.0, group y mean = 1.0.
    assert train_feats["grp_mean"].tolist() == [10.0, 10.0, 1.0, 1.0]
    # Test partition (rows 2,5) computed from ONLY its own rows: each is a
    # singleton group, so the "mean" is just that row's own value -- NOT the
    # 40.0 / 17.33 a globally-fit mean would have produced by mixing train's
    # rows in. The key assertion is that test's value does NOT equal what a
    # global (leaky) computation would have given.
    global_feats = materialize_fold_safe_features(
        df_raw, explanations, np.ones(6, dtype=bool)
    )
    leaky_value_for_row2 = global_feats["grp_mean"].iloc[2]
    assert test_feats["grp_mean"].iloc[0] != leaky_value_for_row2


@pytest.mark.unit
@pytest.mark.config
def test_build_fold_safe_entities_folds_shapes_and_no_id_leak_across_folds():
    """Each fold's dataframe must contain exactly its own ids, with target
    merged in, and total rows must equal the dataset -- a structural regression
    guard for the function that replaces the direct df_entities_attributes slice
    in rogue_one_main.py."""
    df_raw = pd.DataFrame({"id": range(9), "a": range(9)})
    labels_df = pd.DataFrame({"id": range(9), "target": [i % 2 for i in range(9)]})
    explanations = pd.DataFrame(
        [{"Attribute": "a_sq", "Pandas Command": "result = df['a'] ** 2"}]
    )
    fixed_test_ids = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]

    for current in range(3):
        folds = build_fold_safe_entities_folds(
            df_raw=df_raw,
            attribute_explanations=explanations,
            labels_df=labels_df,
            fixed_test_ids=fixed_test_ids,
            current_test_fold_id=current,
        )
        assert len(folds) == 3
        seen_ids = set()
        for fold_id, fold_df in enumerate(folds):
            assert set(fold_df["id"]) == fixed_test_ids[fold_id]
            assert "target" in fold_df.columns
            assert "a_sq" in fold_df.columns
            assert not fold_df["target"].isna().any()
            seen_ids.update(fold_df["id"])
        assert seen_ids == set(range(9))


@pytest.mark.unit
@pytest.mark.config
def test_build_fold_safe_entities_folds_isolates_the_current_test_fold():
    """The fold acting as `current_test_fold_id` must have its aggregate feature
    computed from ONLY its own rows, never from the other (train) folds --
    directly exercising the leak this module exists to close.
    """
    df_raw = pd.DataFrame(
        {
            "id": range(6),
            "grp": ["x", "x", "x", "x", "x", "x"],  # one big group
            "val": [10.0, 10.0, 10.0, 10.0, 10.0, 1000.0],
        }
    )
    labels_df = pd.DataFrame({"id": range(6), "target": [0, 1, 0, 1, 0, 1]})
    explanations = pd.DataFrame(
        [
            {
                "Attribute": "grp_mean",
                "Pandas Command": "result = df.groupby('grp')['val'].transform('mean')",
            }
        ]
    )
    # Fold 2 (ids {4, 5}) is the outlier-containing fold.
    fixed_test_ids = [{0, 1}, {2, 3}, {4, 5}]

    folds_when_fold2_is_test = build_fold_safe_entities_folds(
        df_raw=df_raw,
        attribute_explanations=explanations,
        labels_df=labels_df,
        fixed_test_ids=fixed_test_ids,
        current_test_fold_id=2,
    )
    fold2_df = folds_when_fold2_is_test[2]
    # Fold 2's own rows are val=[10.0, 1000.0] -> its own-partition mean is 505.0,
    # NOT the ~508.3 a leaky global mean (over all 6 rows) would have produced,
    # and NOT the 10.0 the train-side rows would compute either -- each side is
    # self-contained.
    assert fold2_df["grp_mean"].tolist() == [505.0, 505.0]

    # Meanwhile fold 0 (train-role in this call) must be computed from the
    # train-only partition (ids 0-3, all val=10.0), unaffected by fold 2's 1000.0.
    fold0_df = folds_when_fold2_is_test[0]
    assert fold0_df["grp_mean"].tolist() == [10.0, 10.0]


@pytest.mark.unit
@pytest.mark.config
def test_replay_frame_exposes_the_id_column_the_extractor_was_told_to_use():
    """Finding 38 (2026-09-15). At authoring time the Extractor's `df_features` is
    `self.df_entities_attributes`, which always carries the entity-id column, and the
    tool docs steer the model to the canonical
    ``result = df_features['id'].map(df.groupby('id')[...].first())`` shape. Replaying
    that command against a frame with no `id` column raised KeyError and the attribute
    was silently NaN-filled -- 2453 such failures across the 27 seed-42 run
    directories. The replay frame must therefore look like the authoring frame."""
    df_raw = pd.DataFrame({"id": [0, 1, 2, 3], "age": [10.0, 20.0, 30.0, 40.0]})
    explanations = pd.DataFrame(
        [
            {
                "Attribute": "age_by_id",
                "Pandas Command": (
                    "result = df_features['id'].map(df.groupby('id')['age'].first())"
                ),
            }
        ]
    )

    mask = np.array([True, False, True, True])
    out = materialize_fold_safe_features(df_raw, explanations, row_mask=mask)

    assert "id" in out.columns
    assert list(out["id"]) == [0, 2, 3]
    assert not out["age_by_id"].isna().any(), "the canonical idiom must not NaN-fill"
    assert list(out["age_by_id"]) == [10.0, 30.0, 40.0]


@pytest.mark.unit
@pytest.mark.config
def test_length_mismatched_result_nan_fills_instead_of_aborting_the_dataset():
    """Finding 38 (2026-09-15). A command mixing an id-indexed object with a
    positionally-indexed one makes pandas align on the union of the two indexes, so
    `result` comes back LONGER than the partition. That assignment used to sit outside
    the try/except, so the ValueError propagated out of
    `build_fold_safe_entities_folds` and the orchestrator turned it into a whole-dataset
    skip -- six of 27 cells in the 2026-09-11 wave died exactly this way, with rc=0 and
    no fold_aggregation_summary.csv. One NaN column is the documented fallback; losing
    the dataset is not."""
    df_raw = pd.DataFrame({"id": [10, 11, 12, 13], "x": [1.0, 2.0, 3.0, 4.0]})
    explanations = pd.DataFrame(
        [
            {"Attribute": "x_copy", "Pandas Command": "result = df['x']"},
            {
                "Attribute": "misaligned",
                # `first()` is indexed by id (10..13); `x_copy` is indexed 0..n-1, so
                # the division aligns on the union and is longer than the partition.
                "Pandas Command": (
                    "result = df.groupby('id')['x'].first() / df_features['x_copy']"
                ),
            },
            {"Attribute": "after", "Pandas Command": "result = df['x'] * 2"},
        ]
    )

    mask = np.array([True, True, True, False])
    out = materialize_fold_safe_features(df_raw, explanations, row_mask=mask)

    assert len(out) == 3
    assert out["misaligned"].isna().all()
    # The attributes on either side of the bad one must still be computed.
    assert list(out["x_copy"]) == [1.0, 2.0, 3.0]
    assert list(out["after"]) == [2.0, 4.0, 6.0]


@pytest.mark.unit
@pytest.mark.config
def test_build_fold_safe_entities_folds_survives_a_length_mismatched_attribute():
    """The dataset-level guarantee behind the test above: one pathological command
    must not raise out of the fold builder, because the orchestrator reads a ValueError
    from here as 'skip this whole dataset'."""
    df_raw = pd.DataFrame({"id": list(range(10)), "x": [float(i) for i in range(10)]})
    labels_df = pd.DataFrame({"id": list(range(10)), "target": [i % 2 for i in range(10)]})
    explanations = pd.DataFrame(
        [
            {"Attribute": "x_copy", "Pandas Command": "result = df['x']"},
            {
                "Attribute": "misaligned",
                "Pandas Command": (
                    "result = df.groupby('id')['x'].first() / df_features['x_copy']"
                ),
            },
        ]
    )
    fixed_test_ids = [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}]

    folds = build_fold_safe_entities_folds(
        df_raw=df_raw,
        attribute_explanations=explanations,
        labels_df=labels_df,
        fixed_test_ids=fixed_test_ids,
        current_test_fold_id=0,
    )

    assert len(folds) == 5
    for fold in folds:
        assert len(fold) == 2
        assert {"id", "target", "x_copy", "misaligned"} <= set(fold.columns)
