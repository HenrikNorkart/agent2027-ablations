"""Unit tests for train/test splitting utilities."""

import pytest

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.config
def test_train_test_splitter_imports():
    """Verify train_test_splitter can be imported."""
    try:
        from rogueone.utils import train_test_splitter
        assert train_test_splitter is not None
    except ImportError:
        pytest.skip("train_test_splitter not available")


@pytest.mark.unit
@pytest.mark.config
def test_train_test_splitter_determinism(sample_dataframe):
    """
    Test train_test_splitter produces deterministic results with fixed seed.
    
    Per Constitution I (Reproducibility-First): Results must be reproducible with same seed
    Per spec FR-010: Must support deterministic execution
    """
    from rogueone.utils.train_test_splitter import deterministic_kfold_indices

    n_rows = len(sample_dataframe)
    folds_first = deterministic_kfold_indices(n_rows=n_rows, n_splits=5, random_state=42)
    folds_second = deterministic_kfold_indices(n_rows=n_rows, n_splits=5, random_state=42)
    assert folds_first == folds_second


@pytest.mark.unit
@pytest.mark.config
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_train_test_splitter_supports_task_types(sample_dataframe, task_type):
    """
    Test train_test_splitter works for both classification and regression.
    
    Per Constitution IV: task-agnostic design
    """
    _ = task_type
    from rogueone.utils.train_test_splitter import deterministic_kfold_indices

    folds = deterministic_kfold_indices(
        n_rows=len(sample_dataframe),
        n_splits=5,
        random_state=42,
    )
    assert len(folds) == 5


@pytest.mark.unit
@pytest.mark.config
def test_kfold_split_has_no_overlap_and_full_test_coverage(sample_dataframe):
    """Each sample must be in test exactly once across 5 deterministic folds."""
    from rogueone.utils.train_test_splitter import deterministic_kfold_indices

    n_rows = len(sample_dataframe)
    folds = deterministic_kfold_indices(n_rows=n_rows, n_splits=5, random_state=42)

    test_seen: list[int] = []
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        test_seen.extend(test_idx)

    assert sorted(test_seen) == list(range(n_rows))


@pytest.mark.unit
@pytest.mark.config
def test_all_singleton_dedup_groups_do_not_change_the_split():
    """No duplicates -> dedup_groups is a no-op, folds match the plain KFold path.

    Finding 14 (2026-08-25): datasets with no exact-duplicate rows must get exactly
    the same fold assignment as before this fix -- only cmc/diamonds-like datasets
    with real duplicates should ever see a different split.
    """
    import numpy as np

    from rogueone.utils.train_test_splitter import deterministic_kfold_indices

    n_rows = 40
    plain = deterministic_kfold_indices(n_rows=n_rows, n_splits=5, random_state=42)
    all_singleton_groups = np.arange(n_rows)  # every row its own group
    with_groups = deterministic_kfold_indices(
        n_rows=n_rows, n_splits=5, random_state=42, dedup_groups=all_singleton_groups
    )
    assert plain == with_groups


@pytest.mark.unit
@pytest.mark.config
def test_dedup_groups_keeps_duplicate_rows_on_one_side_of_every_fold():
    """A duplicate-row group must never straddle a fold's train/test boundary.

    Regression test for Finding 14 (2026-08-25): `cmc`/`diamonds` have exact
    duplicate rows that a plain KFold happily splits across train/test, letting a
    memorizing evaluator score the held-out twin for free.
    """
    import numpy as np

    from rogueone.utils.train_test_splitter import deterministic_kfold_indices

    n_rows = 40
    # Build 20 duplicate pairs (groups 0..19, each with 2 rows) -- deliberately NOT
    # already fold-aligned, so a naive split would very likely break some of them.
    dedup_groups = np.concatenate([np.arange(20), np.arange(20)])

    folds = deterministic_kfold_indices(
        n_rows=n_rows, n_splits=5, random_state=42, dedup_groups=dedup_groups
    )

    test_seen: list[int] = []
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        train_groups = {dedup_groups[i] for i in train_idx}
        test_groups = {dedup_groups[i] for i in test_idx}
        assert train_groups.isdisjoint(test_groups), (
            "a duplicate-row group was split across train/test"
        )
        test_seen.extend(test_idx)

    assert sorted(test_seen) == list(range(n_rows))


@pytest.mark.unit
@pytest.mark.config
def test_compute_duplicate_groups_matches_exact_row_duplicates():
    """compute_duplicate_groups must group rows that are identical outside id/target."""
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_duplicate_groups

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [1, 1, 2, 1, 3],
            "b": ["x", "x", "y", "x", "z"],
        }
    )
    groups = compute_duplicate_groups(df, exclude_cols=("id",))

    # rows 0, 1, 3 are exact duplicates on (a, b); rows 2 and 4 are each unique.
    assert groups[0] == groups[1] == groups[3]
    assert groups[2] != groups[0]
    assert groups[4] != groups[0]
    assert groups[2] != groups[4]


@pytest.mark.unit
@pytest.mark.config
def test_assert_no_group_split_raises_on_violation():
    """The guard must actually fire when a group is split, not just when asked nicely."""
    import numpy as np

    from rogueone.utils.train_test_splitter import assert_no_group_split

    groups = np.array([0, 0, 1, 2])
    with pytest.raises(ValueError, match="duplicate-row group"):
        assert_no_group_split(train_indices=[0, 2], test_indices=[1, 3], groups=groups)

    # Same group id, but both rows on the same side: must NOT raise.
    assert_no_group_split(train_indices=[0, 1, 2], test_indices=[3], groups=groups)



# ---------------------------------------------------------------------------
# Finding 17 (2026-08-28): the Finding-14 dedup fix was wired to the wrong frame.
# `rogue_one_main` passed `df_entities_attributes`, which is id-only until the
# Extractor appends features, so every row keyed to "" and the dataset collapsed into
# a single group. `deterministic_kfold_indices` then refused to build folds and the
# run exited rc=0 with no result -- silently, for every dataset it was applied to.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_duplicate_groups_rejects_frame_with_no_comparable_columns():
    """An id-only frame must raise, not collapse into one group."""
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_duplicate_groups

    df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="at least one column to compare"):
        compute_duplicate_groups(df, exclude_cols=("id",))


@pytest.mark.unit
def test_entity_groups_one_row_per_entity_matches_row_level_grouping():
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_entity_duplicate_groups

    df = pd.DataFrame(
        {"id": [10, 11, 12, 13], "a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]}
    )
    groups = compute_entity_duplicate_groups(df, [10, 11, 12, 13], exclude_cols=("id",))
    assert groups[0] == groups[1]  # entities 10 and 11 are duplicates
    assert len({groups[0], groups[2], groups[3]}) == 3


@pytest.mark.unit
def test_entity_groups_follow_the_requested_entity_order():
    """Groups are positional: they must line up with the feature frame's id order."""
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_entity_duplicate_groups

    df = pd.DataFrame({"id": [1, 2, 3], "a": [7, 8, 7]})
    groups = compute_entity_duplicate_groups(df, [2, 3, 1], exclude_cols=("id",))
    # In the order (2, 3, 1): entity 2 is unique, entities 3 and 1 are the duplicates.
    assert groups[1] == groups[2]
    assert groups[0] != groups[1]


@pytest.mark.unit
def test_entity_groups_multi_row_entities_ignore_row_order():
    """Two entities whose raw rows match as a multiset are duplicates."""
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_entity_duplicate_groups

    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "a": [5, 6, 6, 5, 9, 9]})
    groups = compute_entity_duplicate_groups(df, [1, 2, 3], exclude_cols=("id",))
    assert groups[0] == groups[1]  # same rows, different order
    assert groups[2] != groups[0]


@pytest.mark.unit
def test_entity_groups_produce_usable_folds_on_a_cmc_shaped_frame():
    """The end-to-end regression: this is exactly what silently skipped every run.

    A low-cardinality table with real duplicates must still yield 5 folds. Under the
    old wiring this raised "only 1 distinct duplicate groups for N rows".
    """
    import numpy as np
    import pandas as pd

    from rogueone.utils.train_test_splitter import (
        assert_no_group_split,
        compute_entity_duplicate_groups,
        deterministic_kfold_indices,
    )

    n = 300
    df = pd.DataFrame(
        {
            "id": list(range(n)),
            "a": [i % 40 for i in range(n)],
            "b": [i % 7 for i in range(n)],
        }
    )
    entity_ids = df["id"].tolist()
    groups = compute_entity_duplicate_groups(df, entity_ids, exclude_cols=("id",))
    assert len(np.unique(groups)) > 5, "must not collapse into a single group"

    folds = deterministic_kfold_indices(
        n_rows=len(entity_ids),
        n_splits=5,
        random_state=42,
        dedup_groups=groups,
    )
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert test_idx, "no fold may be empty"
        assert_no_group_split(train_idx, test_idx, groups)


@pytest.mark.unit
@pytest.mark.config
def test_compute_duplicate_groups_handles_missing_values():
    """A frame with NaN must still group, rather than raising inside the key join.

    Regression test for 2026-09-11. Under pandas 3's default string dtype,
    `df.astype(str)` leaves a missing value as a float nan instead of the string "nan",
    so building the row key with `.agg("|".join, axis=1)` raised
    `TypeError: sequence item N: expected str instance, float found`. The orchestrator
    caught that, printed "Error running experiment", and exited **rc=0** with no result --
    so it read as a success. It affected 4 of the 27 reported datasets (arrhythmia,
    breast-w, eucalyptus, housing) the first time this code ran end-to-end.
    """
    import numpy as np
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_duplicate_groups

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1.0, 1.0, np.nan, np.nan],
            "b": ["x", "x", "y", "y"],
        }
    )
    groups = compute_duplicate_groups(df, exclude_cols=("id",))

    # Rows 0 and 1 are identical; rows 2 and 3 are identical to each other INCLUDING the
    # hole -- two rows missing the same field are duplicates, not distinct.
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


@pytest.mark.unit
@pytest.mark.config
def test_missing_value_does_not_collide_with_its_own_string_spelling():
    """A real "nan"/"NA" string must not be grouped with an actual missing value.

    The obvious fix for the bug above -- letting NaN render as the string "nan" -- would
    silently merge a row whose value is genuinely the text "nan" with a row that has no
    value at all. The sentinel exists so that cannot happen.
    """
    import numpy as np
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_duplicate_groups

    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "a": ["nan", np.nan, "NA"],
        }
    )
    groups = compute_duplicate_groups(df, exclude_cols=("id",))

    assert len({groups[0], groups[1], groups[2]}) == 3


@pytest.mark.unit
@pytest.mark.config
def test_entity_groups_handle_missing_values():
    """The entity-level grouper has the same join and needs the same guarantee."""
    import numpy as np
    import pandas as pd

    from rogueone.utils.train_test_splitter import compute_entity_duplicate_groups

    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "a": [np.nan, 2.0, np.nan, 2.0, 5.0],
        }
    )
    groups = compute_entity_duplicate_groups(df, entity_ids=[1, 2, 3], exclude_cols=("id",))

    # Entities 1 and 2 have the same multiset of rows, holes included; 3 does not.
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
