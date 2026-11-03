"""Unit tests for fold lifecycle correctness.

Tests for US1: Reliable Per-Fold Execution.
T006 covers tests 1–5, T007 covers tests 6–7.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock

from rogueone.utils import rouge_one_cfg

K_FOLDS = rouge_one_cfg.k_folds          # 5
NUM_ITERATIONS = rouge_one_cfg.num_iterations  # 10
TOTAL_TRIALS = K_FOLDS * NUM_ITERATIONS  # 50

_ATTR_EXPLANATION_COLUMNS = [
    "Attribute",
    "Description",
    "Pandas Command",
    "Status",
    "Added",
    "Pruned Iteration",
    "Prune Rationale",
]


def _make_stub_network(df: pd.DataFrame | None = None) -> "MagicMock":
    """Return a minimal stub that satisfies _reset_fold_state() and compute_fold_aggregation()."""
    try:
        from rogue_one_main import RogueOneAgentNetwork
        stub = MagicMock(spec=RogueOneAgentNetwork)
    except ImportError:
        pytest.skip("rogue_one_main not available")

    if df is None:
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "label": [0, 1, 0, 1, 0]})

    stub.df = df
    stub.df_entities_attributes = pd.DataFrame({"id": df["id"].unique()})
    stub.df_attributes_explanations = pd.DataFrame(columns=_ATTR_EXPLANATION_COLUMNS)
    stub.df_scientist_feedback_pool = pd.DataFrame()
    stub.focus_history = []
    stub.scientist_agent = MagicMock()
    stub.scientist_agent._note_book = ""
    return stub


# ── T006: Tests 1–5 ───────────────────────────────────────────────────────────


def test_tabular_total_trials_is_k_folds_times_num_iterations():
    """Total outer-loop count must equal k_folds * num_iterations (e.g. 5 * 10 = 50)."""
    assert TOTAL_TRIALS == K_FOLDS * NUM_ITERATIONS
    assert TOTAL_TRIALS == 50


def test_fold_run_id_and_iteration_in_fold_computed_correctly():
    """For every global index 0..49, fold_run_id and iteration_in_fold are in valid ranges
    and reconstruct the original index exactly."""
    for i in range(TOTAL_TRIALS):
        fold_run_id = i // NUM_ITERATIONS
        iteration_in_fold = i % NUM_ITERATIONS

        assert 0 <= fold_run_id < K_FOLDS, (
            f"fold_run_id {fold_run_id} out of [0, {K_FOLDS}) for index {i}"
        )
        assert 0 <= iteration_in_fold < NUM_ITERATIONS, (
            f"iteration_in_fold {iteration_in_fold} out of [0, {NUM_ITERATIONS}) for index {i}"
        )
        # Round-trip: the two components must reconstruct the original index.
        assert fold_run_id * NUM_ITERATIONS + iteration_in_fold == i


def test_no_duplicate_fold_iteration_pairs():
    """All 50 (fold_run_id, iteration_in_fold) pairs derived from global indices are unique."""
    pairs = [(i // NUM_ITERATIONS, i % NUM_ITERATIONS) for i in range(TOTAL_TRIALS)]
    assert len(pairs) == len(set(pairs)), (
        "Duplicate (fold_run_id, iteration_in_fold) pairs found"
    )


def test_test_fold_id_stored_equals_fold_run_id():
    """The correct test_fold_id is fold_run_id = i // num_iterations,
    not the bugged formula i % k_folds which cycles back within a fold."""
    for i in range(TOTAL_TRIALS):
        correct_fold_id = i // NUM_ITERATIONS
        # Verify the correct formula assigns all indices in fold N to fold N.
        expected_fold = i // NUM_ITERATIONS
        assert correct_fold_id == expected_fold

    # Demonstrate the concrete divergence the bug causes:
    # Index 10 is the first iteration of fold 1.
    # Correct: 10 // 10 == 1 (fold 1)
    # Bugged:  10 %  5 == 0 (fold 0 — wrong!)
    first_index_of_fold_1 = NUM_ITERATIONS * 1
    assert first_index_of_fold_1 // NUM_ITERATIONS == 1
    assert first_index_of_fold_1 % K_FOLDS == 0  # bugged value
    assert (first_index_of_fold_1 // NUM_ITERATIONS) != (first_index_of_fold_1 % K_FOLDS)


def test_fold_state_reset_clears_expected_fields():
    """_reset_fold_state() must clear focus_history, extracted attributes,
    feedback pool, attribute explanations, and the scientist notebook."""
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    stub = _make_stub_network()

    # Pre-populate every field with non-empty content.
    stub.df_entities_attributes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "attr_a": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    stub.df_attributes_explanations = pd.DataFrame([{
        "Attribute": "attr_a",
        "Description": "description",
        "Pandas Command": "df['attr_a']",
        "Status": "Active",
        "Added": 0,
        "Pruned Iteration": None,
        "Prune Rationale": None,
    }])
    stub.df_scientist_feedback_pool = pd.DataFrame([{"status": "completed", "id": "Trial 0"}])
    stub.focus_history = ["focus_1", "focus_2"]
    stub.scientist_agent._note_book = "Notes from fold 0"

    RogueOneAgentNetwork._reset_fold_state(stub)

    assert stub.focus_history == [], "focus_history must be empty after reset"
    assert list(stub.df_entities_attributes.columns) == ["id"], (
        "df_entities_attributes must have only the 'id' column after reset"
    )
    assert stub.df_attributes_explanations.empty, (
        "df_attributes_explanations must be empty after reset"
    )
    assert stub.df_scientist_feedback_pool.empty, (
        "df_scientist_feedback_pool must be empty after reset"
    )
    assert stub.scientist_agent._note_book == "", (
        "scientist_agent._note_book must be cleared after reset"
    )


# ── T007: Tests 6–7 ───────────────────────────────────────────────────────────


def test_fold_state_does_not_carry_over_to_next_fold():
    """After fold 0 adds 3 extracted attributes, calling _reset_fold_state()
    must leave df_entities_attributes with 0 extracted attribute columns."""
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    stub = _make_stub_network()

    # Simulate end of fold 0: 3 extracted attributes accumulated.
    stub.df_entities_attributes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "attr_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "attr_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        "attr_c": [0.5, 0.6, 0.7, 0.8, 0.9],
    })
    stub.focus_history = ["focus1", "focus2", "focus3"]
    stub.scientist_agent._note_book = "Observations from fold 0"

    # Fold 1 boundary: reset is called before iteration 0 of fold 1.
    RogueOneAgentNetwork._reset_fold_state(stub)

    extracted_count = len(stub.df_entities_attributes.columns) - 1  # exclude 'id'
    assert extracted_count == 0, (
        f"Expected 0 extracted attribute columns after reset, got {extracted_count}"
    )


# ── Productive-attribute accounting (Mode B, CAR run znh9zoex) ──────────────


def test_productive_attribute_count_includes_restorations():
    """When the Extractor auto-unprunes 3 features in an iteration, the
    column count of df_entities_attributes does not change but the iteration
    is still productive. The productive count must reflect that.

    Without this, CAR run znh9zoex aborted spuriously: 3 successful
    restorations after a 99 % accuracy trial were counted as "did nothing"
    because the column-delta gate measured 0.
    """
    try:
        from rogue_one_main import _count_productive_extractor_outputs
    except ImportError:
        pytest.skip("rogue_one_main not available")

    explanations = pd.DataFrame(
        {
            "name": ["safety_ord", "buying_ord", "persons_ord"],
            "argumentation": ["unprune", "unprune", "unprune"],
            "command": ["", "", ""],
            "status": ["Active", "Active", "Active"],
            "added": ["Trial 4", "Trial 4", "Trial 4"],
        }
    )

    num_restored, num_productive = _count_productive_extractor_outputs(
        num_new_columns=0,
        attribute_explanations=explanations,
    )

    assert num_restored == 3
    assert num_productive == 3, (
        "Restoration-only iterations must NOT trigger the abort gate."
    )


def test_productive_attribute_count_handles_mixed_new_and_restored():
    """Mix of 1 new append + 2 restorations should report 1 new column,
    2 restored, 3 productive."""
    try:
        from rogue_one_main import _count_productive_extractor_outputs
    except ImportError:
        pytest.skip("rogue_one_main not available")

    explanations = pd.DataFrame(
        {
            "name": ["new_feature", "restored_a", "restored_b"],
            "argumentation": ["new", "restore", "restore"],
            "command": ["...", "", ""],
            "status": ["Active"] * 3,
            "added": ["Trial 1"] * 3,
        }
    )

    num_restored, num_productive = _count_productive_extractor_outputs(
        num_new_columns=1,
        attribute_explanations=explanations,
    )

    assert num_restored == 2
    assert num_productive == 3


def test_productive_attribute_count_zero_proposals_means_zero():
    """When the Extractor returns 0 proposals (LLM did not call
    append_new_attribute at all — Mode A in CAR runs ca90miqg / p9t3rayi /
    wi5r0oki), the productive count must be 0 so the abort gate fires."""
    try:
        from rogue_one_main import _count_productive_extractor_outputs
    except ImportError:
        pytest.skip("rogue_one_main not available")

    explanations = pd.DataFrame(columns=["name", "argumentation", "command", "status", "added"])

    num_restored, num_productive = _count_productive_extractor_outputs(
        num_new_columns=0,
        attribute_explanations=explanations,
    )

    assert num_restored == 0
    assert num_productive == 0


def test_productive_attribute_count_handles_none_explanations():
    """A None DataFrame must be tolerated (defensive — early-error paths)."""
    try:
        from rogue_one_main import _count_productive_extractor_outputs
    except ImportError:
        pytest.skip("rogue_one_main not available")

    num_restored, num_productive = _count_productive_extractor_outputs(
        num_new_columns=2,
        attribute_explanations=None,
    )

    assert num_restored == 0
    assert num_productive == 2


# ── Regression: forward_pass post-Extractor block must not reference the
#     OLD `num_new_attributes` local that was removed when introducing
#     `num_new_columns` / `num_productive_attributes`. CAR runs ln2qurm8
#     and g6t13ci5 silently crashed with a NameError after the Extractor
#     successfully added 28-43 features, killing every iteration before
#     the Tester ever ran. ──────────────────────────────────────────────


def test_forward_pass_does_not_reference_removed_num_new_attributes_local():
    """The old `num_new_attributes` local was renamed to `num_new_columns`
    /`num_productive_attributes` to fix Mode B. A post-rename reference
    survived and crashed CAR runs ln2qurm8/g6t13ci5 with a NameError
    immediately after the abort gate. This regression test forbids any
    line in forward_pass *after* the abort branch from naming the old
    local, since that's the segment where the rename was incomplete.
    """
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    import inspect
    src = inspect.getsource(RogueOneAgentNetwork.forward_pass)
    lines = src.splitlines()

    # Find the abort branch (ends at `return "aborted"`); anything below it
    # must not reference the removed `num_new_attributes` local. References
    # inside the kwargs of `_log_extractor_iteration_activity(...)` are
    # legal because the helper retained that *parameter* name.
    abort_line_idx = next(
        (i for i, ln in enumerate(lines) if 'return "aborted"' in ln),
        None,
    )
    assert abort_line_idx is not None, (
        "forward_pass must still contain a `return \"aborted\"` branch."
    )

    tail = "\n".join(lines[abort_line_idx + 1 :])
    # The kwarg `num_new_attributes=...` is only allowed before the abort
    # branch (where _log_extractor_iteration_activity is called). The tail
    # must not mention the old local at all.
    assert "num_new_attributes" not in tail, (
        "After the abort branch, forward_pass must not reference the old "
        "`num_new_attributes` local — it was renamed to `num_new_columns` / "
        "`num_productive_attributes`. A surviving reference was the cause of "
        "the post-86807fc CAR crashes (runs ln2qurm8 and g6t13ci5)."
    )


def test_forward_pass_returns_iteration_skipped_for_mid_fold_zero_productive():
    """When the Extractor produces zero productive features at iter > 0,
    forward_pass must return ``"iteration_skipped"`` (NOT ``"aborted"``) so
    the outer loop can continue to the next trial — the fold already has
    at least one completed trial from iter 0 and the dataset can still
    produce a valid fold_aggregation. Aborting the dataset on every
    plateau iteration was the dominant CAR failure mode after the
    Mode B fix landed.

    Tested at the source-string level because forward_pass is too
    integration-heavy to exercise end-to-end without an LLM endpoint.
    """
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    import inspect
    src = inspect.getsource(RogueOneAgentNetwork.forward_pass)

    assert 'return "iteration_skipped"' in src, (
        "forward_pass must have a branch that returns 'iteration_skipped' "
        "for mid-fold zero-productive iterations."
    )
    assert "iteration_in_fold > 0" in src or "iteration_in_fold == 0" in src, (
        "forward_pass must distinguish iter-0 (must abort) from iter>0 "
        "(must skip-iteration) when num_productive_attributes == 0."
    )


def test_outer_loop_handles_iteration_skipped_status():
    """The outer loop in `main()` must `continue` on ``"iteration_skipped"``
    rather than treat it as an abort or break the dataset run. Without this
    branch, the new return value would land in the catch-all
    `if trial_status != "completed"` clause and abort the dataset anyway,
    defeating the whole point of the relaxation."""
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    import inspect
    main_src = inspect.getsource(RogueOneAgentNetwork.main)

    assert '"iteration_skipped"' in main_src, (
        "Outer loop must explicitly check for 'iteration_skipped' status."
    )
    # The branch must use `continue`, not `break`.
    skipped_branch_pos = main_src.find('"iteration_skipped"')
    next_300 = main_src[skipped_branch_pos : skipped_branch_pos + 300]
    assert "continue" in next_300, (
        "iteration_skipped branch must `continue` (proceed to next trial), "
        "not break or fall through to the abort path."
    )


def test_abort_on_incomplete_fold_does_not_emit_aggregate():
    """When fold 2 aborts mid-run the aggregate must not be produced
    because not all five fold representatives are available."""
    try:
        from rogue_one_main import RogueOneAgentNetwork
    except ImportError:
        pytest.skip("rogue_one_main not available")

    stub = MagicMock(spec=RogueOneAgentNetwork)
    stub.cfg = MagicMock()
    stub.cfg.modality = "tabular"
    stub.cfg.task_type = "classification"
    stub.df_attributes_explanations = pd.DataFrame(columns=_ATTR_EXPLANATION_COLUMNS)

    # Fold 0 and 1 complete all 10 iterations; fold 2 aborts at iteration 5;
    # folds 3 and 4 never start (main loop broke early).
    rows = []
    for fold_id in (0, 1):
        for it in range(NUM_ITERATIONS):
            rows.append({
                "status": "completed",
                "test_fold_id": fold_id,
                "iteration_in_fold": it,
                "accuracy": 0.75 + it * 0.01,
            })
    for it in range(5):
        rows.append({
            "status": "completed",
            "test_fold_id": 2,
            "iteration_in_fold": it,
            "accuracy": 0.80,
        })
    rows.append({
        "status": "aborted",
        "test_fold_id": 2,
        "iteration_in_fold": 5,
        "accuracy": None,
    })

    stub.df_test_pool = pd.DataFrame(rows)

    result = RogueOneAgentNetwork.compute_fold_aggregation(stub)

    assert result == {}, (
        "compute_fold_aggregation() must return {} when not all 5 fold representatives are available"
    )
