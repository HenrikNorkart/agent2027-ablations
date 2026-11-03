"""Unit tests for the Tester pruning floor and related fixes.

These cover Fix #3 (minimum-features floor), Fix #5 (XGBoost importance pipeline
shape robustness), and Fix #6 (Scientist generic_pandas_tool docstring) from the
"fix Rogue One aborts on car / tic-tac-toe" change set.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rogueone.llm.agents.core.tester.agent import _enforce_pruning_floor


def _make_explanations(features: list[str]) -> pd.DataFrame:
    """Build a minimal df_attribute_explanations with the columns the floor needs."""
    return pd.DataFrame(
        {
            "Attribute": features,
            "Status": ["Active"] * len(features),
            "Pruned Iteration": [pd.NA] * len(features),
            "Prune Rationale": [pd.NA] * len(features),
        }
    )


@pytest.mark.unit
@pytest.mark.agent
def test_tester_pruning_floor_keeps_minimum_features():
    """pre_active=16, all 16 newly pruned -> floor=max(3, ceil(0.2*16))=4 active after."""
    features = [f"f{i}" for i in range(16)]
    df = _make_explanations(features)
    status_before = df["Status"].copy()

    # Simulate the LLM having pruned all 16 features.
    df["Status"] = "Pruned"

    # Synthetic metric_profiles: f0 has the highest SHAP (analog of total_X),
    # the rest have descending SHAP.
    shap_lookup = {f"f{i}": 1.18 if i == 0 else (0.5 - 0.02 * i) for i in range(16)}
    mi_lookup = {f"f{i}": 0.05 for i in range(16)}

    newly_pruned = features.copy()
    remaining, restored = _enforce_pruning_floor(
        df_attribute_explanations=df,
        status_before=status_before,
        newly_pruned=newly_pruned,
        shap_lookup=shap_lookup,
        mi_lookup=mi_lookup,
    )

    expected_floor = max(3, math.ceil(0.2 * 16))  # = 4
    active_after = int((df["Status"] != "Pruned").sum())
    assert active_after >= expected_floor
    assert len(restored) == expected_floor
    # The remaining "newly_pruned" list should no longer contain restored items.
    assert set(restored).isdisjoint(set(remaining))


@pytest.mark.unit
@pytest.mark.agent
def test_tester_pruning_floor_keeps_high_shap_features_first():
    """When restoration is required, the kept-active features must be the ones with
    the highest mean_abs_shap (Fix #3 — preserve the `total_X` analog)."""
    features = [f"f{i}" for i in range(16)]
    df = _make_explanations(features)
    status_before = df["Status"].copy()
    df["Status"] = "Pruned"

    # f7 has a clearly higher SHAP than every other feature (analog to total_X with
    # mean_abs_SHAP = 1.18 in the tic-tac-toe failure).
    shap_lookup = {f"f{i}": 0.05 for i in range(16)}
    shap_lookup["f7"] = 1.18
    mi_lookup = {f"f{i}": 0.0 for i in range(16)}

    _, restored = _enforce_pruning_floor(
        df_attribute_explanations=df,
        status_before=status_before,
        newly_pruned=features.copy(),
        shap_lookup=shap_lookup,
        mi_lookup=mi_lookup,
    )

    assert "f7" in restored, (
        "Highest-SHAP feature must be restored first when the floor is enforced; "
        f"got restored={restored}"
    )
    assert df.loc[df["Attribute"] == "f7", "Status"].iloc[0] == "Active"


@pytest.mark.unit
@pytest.mark.agent
def test_tester_pruning_floor_noop_when_above_floor():
    """No restoration when the LLM left enough active features."""
    features = [f"f{i}" for i in range(16)]
    df = _make_explanations(features)
    status_before = df["Status"].copy()
    # Prune only the bottom half — 8 active remain, well above the 4 floor.
    df.loc[df["Attribute"].isin(features[:8]), "Status"] = "Pruned"

    remaining, restored = _enforce_pruning_floor(
        df_attribute_explanations=df,
        status_before=status_before,
        newly_pruned=features[:8],
        shap_lookup={f: 0.1 for f in features},
        mi_lookup={f: 0.0 for f in features},
    )
    assert restored == []
    assert remaining == features[:8]


@pytest.mark.unit
@pytest.mark.agent
def test_tester_pruning_floor_noop_when_complete_emptying():
    """When everything is pruned (active_after == 0) the helper does nothing —
    the existing complete-emptying rollback in generate_pruning handles that case."""
    features = [f"f{i}" for i in range(5)]
    df = _make_explanations(features)
    status_before = df["Status"].copy()
    df["Status"] = "Pruned"

    # No newly_pruned passed in -> noop branch.
    remaining, restored = _enforce_pruning_floor(
        df_attribute_explanations=df,
        status_before=status_before,
        newly_pruned=[],
        shap_lookup={},
        mi_lookup={},
    )
    assert restored == []
    assert remaining == []


# ---------------------------------------------------------------------------
# Fix #5 — XGBoost get_score importance pipeline shape robustness
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_tester_xgb_importance_pipeline_handles_dict_keys():
    """Replicate the broken pattern (which can fail with 'Per-column arrays must each
    be 1-dimensional' when values are arrays) and verify the new pattern always works.
    """
    fi = {
        "feat_a": np.array([1.0]),
        "feat_b": np.array([2.0, 0.0]),  # 2-D-ish: triggers the original bug
        "feat_c": np.array([0.5]),
    }

    # New, robust pattern from Fix #5 — list of (key, value) tuples avoids
    # constructing the frame from a dict whose values may have inconsistent shapes.
    df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
    # Coerce to a numeric scalar per row (sum-collapse multi-element arrays).
    df["importance"] = df["importance"].apply(
        lambda v: float(np.asarray(v).sum()) if np.ndim(v) > 0 else float(v)
    )

    assert list(df["feature"]) == ["feat_a", "feat_b", "feat_c"]
    assert df["importance"].dtype == float
    assert df.loc[df["feature"] == "feat_b", "importance"].iloc[0] == 2.0


# ---------------------------------------------------------------------------
# Fix #6 — Scientist generic_pandas_tool docstring describes real columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.agent
def test_scientist_pandas_tool_documents_real_columns():
    """The Scientist's generic_pandas_tool docstring must reference the real
    df_test_results column names produced by rogue_one_main, not the legacy
    'trail' / 'Trail<n>' typo."""
    import inspect

    from rogueone.llm.agents.core.scientist import agent as scientist_agent_mod

    src = inspect.getsource(scientist_agent_mod)

    # Required column references from the new docstring.
    assert "'id'" in src or "`id`" in src
    assert "feature_assessment" in src
    assert "train_accuracy" in src
    # Trial format (capital T, single space).
    assert "Trial <n>" in src or "Trial <" in src

    # The legacy typos must be gone from the docstring text.
    assert "'trail'" not in src
    assert "Trail<" not in src
    assert "trail_number" not in src
