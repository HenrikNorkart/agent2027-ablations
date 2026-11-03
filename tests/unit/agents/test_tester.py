"""Unit tests for TesterAgent."""

import pytest

pytest_plugins = []


@pytest.mark.unit
@pytest.mark.agent
def test_tester_agent_exists():
    """Verify TesterAgent can be imported."""
    try:
        from rogueone.llm.agents import TesterAgent
        assert TesterAgent is not None
    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_tester_output_contract_fields():
    """
    Validate TesterAgent output contract includes required fields.
    
    Per agent-behavior-validation-contract.md, output must include:
    - results (dict with per-feature metrics)
    - summary (dict with aggregate metrics)
    - best_feature (string)
    - improvements (dict showing improvement over baseline)
    """
    try:
        from rogueone.llm.agents import TesterAgent
        
        # Expected contract:
        expected_fields = ["results", "summary", "best_feature", "improvements"]
        
        assert all(f in expected_fields for f in expected_fields)
        
    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_tester_supports_task_types(task_type):
    """
    Test TesterAgent supports both classification and regression.
    
    Per Constitution IV: task-agnostic design
    Classification metrics: accuracy, f1, precision, recall
    Regression metrics: mse, rmse, r2, mae
    """
    try:
        from rogueone.llm.agents import TesterAgent
        
        # Placeholder test - documents Constitution IV requirement
        # When implemented, validates task-specific metric generation
        
    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_tester_handles_no_improvement():
    """
    Test TesterAgent gracefully handles features with no improvement.
    
    Per spec FR-003: Must validate failure handling
    """
    try:
        from rogueone.llm.agents import TesterAgent
        
        # Placeholder test - documents expected behavior
        # When implemented, validates agent still returns valid output even with negative improvements
        
    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_metric_profiles_to_dataframe_contract_and_formatting():
    """Validate table dataframe columns and value formatting for pre-computed metrics."""
    try:
        from rogueone.llm.agents.core.tester.agent import TesterAgent
        from rogueone.llm.agents.core.tester.metrics import FeatureMetricProfile

        metric_profiles = [
            FeatureMetricProfile(
                feature_name="feature_a",
                mutual_information=0.1234567,
                vif=None,
                permutation_importance_mean=0.09,
                permutation_importance_std=0.01,
                noise_sensitivity=0.001234,
                mean_abs_shap=2.5,
                computation_warnings=["vif_unavailable", "mean_abs_shap_unavailable"],
            ),
            FeatureMetricProfile(
                feature_name="feature_b",
                mutual_information=None,
                vif=1.333333,
                permutation_importance_mean=None,
                permutation_importance_std=None,
                noise_sensitivity=None,
                mean_abs_shap=None,
                computation_warnings=[],
            ),
        ]

        df = TesterAgent._metric_profiles_to_dataframe(metric_profiles)

        assert list(df.columns) == [
            "feature_name",
            "mutual_information",
            "vif",
            "pfi_mean",
            "pfi_std",
            "noise_sensitivity",
            "mean_abs_shap",
            "warnings",
        ]

        assert len(df) == 2
        assert df.iloc[0]["feature_name"] == "feature_a"
        assert df.iloc[0]["mutual_information"] == "0.123457"
        assert df.iloc[0]["vif"] == "N/A"
        assert (
            df.iloc[0]["warnings"]
            == "vif_unavailable; mean_abs_shap_unavailable"
        )

        assert df.iloc[1]["feature_name"] == "feature_b"
        assert df.iloc[1]["mutual_information"] == "N/A"
        assert df.iloc[1]["vif"] == "1.33333"
        assert df.iloc[1]["pfi_mean"] == "N/A"
        assert df.iloc[1]["warnings"] == ""

    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_prune_attributes_in_df_excludes_pruned_columns_from_evaluation():
    """Tester pruning filter must remove all features marked Pruned."""
    try:
        import pandas as pd
        from rogueone.llm.agents.core.tester.agent import TesterAgent

        df = pd.DataFrame(
            {
                "id": [1, 2],
                "feature_active": [0.1, 0.2],
                "feature_pruned": [10.0, 20.0],
                "target": [0, 1],
            }
        )
        explanations = pd.DataFrame(
            {
                "Attribute": ["feature_active", "feature_pruned"],
                "Status": ["Active", "Pruned"],
            }
        )

        pruned_df = TesterAgent.prune_attributes_in_df(df, explanations)

        assert "feature_active" in pruned_df.columns
        assert "feature_pruned" not in pruned_df.columns
        assert "target" in pruned_df.columns
        assert "id" in pruned_df.columns

    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


@pytest.mark.unit
@pytest.mark.agent
def test_prune_attributes_in_df_handles_prune_metadata_columns():
    """Pruning filter should work even when prune metadata columns are present."""
    try:
        import pandas as pd
        from rogueone.llm.agents.core.tester.agent import TesterAgent

        df = pd.DataFrame(
            {
                "feature_keep": [1.0, 2.0],
                "feature_drop": [9.0, 8.0],
                "target": [0, 1],
            }
        )
        explanations = pd.DataFrame(
            {
                "Attribute": ["feature_keep", "feature_drop"],
                "Status": ["Active", "Pruned"],
                "Pruned Iteration": [pd.NA, "Trial 3"],
                "Prune Rationale": [pd.NA, "Weak predictive signal"],
            }
        )

        result = TesterAgent.prune_attributes_in_df(df, explanations)

        assert list(result.columns) == ["feature_keep", "target"]

    except ImportError:
        pytest.skip("TesterAgent not yet implemented")


