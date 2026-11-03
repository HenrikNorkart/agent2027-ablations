"""Integration tests for orchestration entrypoint."""

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

pytest_plugins = []


@pytest.mark.integration
def test_rogue_one_main_importable():
    """
    Test rogue_one_main module can be imported.

    Per spec FR-005: Must validate orchestration entrypoint
    """
    try:
        import rogue_one_main

        assert rogue_one_main is not None
    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_orchestration_with_classification_task():
    """
    Test orchestration works with classification task.

    Per spec FR-005 + Constitution IV: Must support both task types
    """
    try:
        import rogue_one_main

        # Placeholder test - documents expected behavior
        # When implemented, validates execution succeeds with classification config

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_orchestration_with_regression_task():
    """
    Test orchestration works with regression task.

    Per spec FR-005 + Constitution IV: Must support both task types
    """
    try:
        import rogue_one_main

        # Placeholder test - documents expected behavior
        # When implemented, validates execution succeeds with regression config

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_orchestration_produces_results():
    """
    Test orchestration produces expected output.

    Per spec FR-009: Must validate test outputs clearly identify functional areas
    """
    try:
        import rogue_one_main

        # Placeholder test - documents expected output validation
        # When implemented, validates orchestration returns expected result format

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_update_attribute_pool_keeps_pruned_definition_rows():
    """Pruned features must remain in metadata with Status=Pruned for later iterations."""
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Active",
                    "Added": "Trial 0",
                }
            ]
        )
        network.df_test_pool = None

        incoming = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Pruned",
                    "Added": "Trial 0",
                }
            ]
        )

        asyncio.run(network.update_attribute_pool(incoming, source_agent="tester"))

        assert len(network.df_attributes_explanations) == 1
        assert network.df_attributes_explanations.iloc[0]["Attribute"] == "f_a"
        assert network.df_attributes_explanations.iloc[0]["Status"] == "Pruned"

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_update_attribute_pool_allows_unprune_from_extractor():
    """A previously pruned feature can be restored to Active when later evidence supports it."""
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Pruned",
                    "Added": "Trial 0",
                }
            ]
        )
        network.df_test_pool = None

        incoming = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Active",
                    "Added": "Trial 2",
                }
            ]
        )

        asyncio.run(network.update_attribute_pool(incoming, source_agent="extractor"))

        assert len(network.df_attributes_explanations) == 1
        assert network.df_attributes_explanations.iloc[0]["Status"] == "Active"

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_update_attribute_pool_blocks_unprune_from_tester():
    """Tester-origin updates must not reactivate a pruned feature."""
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Pruned",
                    "Added": "Trial 0",
                }
            ]
        )
        network.df_test_pool = None

        incoming = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Active",
                    "Added": "Trial 2",
                }
            ]
        )

        asyncio.run(network.update_attribute_pool(incoming, source_agent="tester"))

        assert len(network.df_attributes_explanations) == 1
        assert network.df_attributes_explanations.iloc[0]["Status"] == "Pruned"

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_update_attribute_pool_persists_prune_metadata_columns():
    """Prune metadata fields should be preserved in the shared attribute pool."""
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Active",
                    "Added": "Trial 0",
                    "Pruned Iteration": pd.NA,
                    "Prune Rationale": pd.NA,
                }
            ]
        )
        network.df_test_pool = None

        incoming = pd.DataFrame(
            [
                {
                    "Attribute": "f_a",
                    "Description": "feature a",
                    "Pandas Command": "result = ...",
                    "Status": "Pruned",
                    "Added": "Trial 0",
                    "Pruned Iteration": "Trial 2",
                    "Prune Rationale": "Low mutual information and high redundancy.",
                }
            ]
        )

        asyncio.run(network.update_attribute_pool(incoming, source_agent="tester"))

        row = network.df_attributes_explanations.iloc[0]
        assert row["Status"] == "Pruned"
        assert row["Pruned Iteration"] == "Trial 2"
        assert row["Prune Rationale"] == "Low mutual information and high redundancy."

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_compute_fold_aggregation_classification_uses_best_accuracy_per_fold():
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.cfg = SimpleNamespace(
            modality="tabular",
            task_type="classification",
            experiment_name="demo-classification",
        )
        network.df_test_pool = pd.DataFrame(
            [
                {
                    "id": "Trial 0A",
                    "status": "completed",
                    "test_fold_id": 0,
                    "accuracy": 0.71,
                    "precision": 0.70,
                    "recall": 0.69,
                    "f1": 0.68,
                    "AUROC": 0.73,
                },
                {
                    "id": "Trial 0B",
                    "status": "completed",
                    "test_fold_id": 0,
                    "accuracy": 0.82,
                    "precision": 0.81,
                    "recall": 0.80,
                    "f1": 0.79,
                    "AUROC": 0.84,
                },
                {
                    "id": "Trial 1A",
                    "status": "completed",
                    "test_fold_id": 1,
                    "accuracy": 0.76,
                    "precision": 0.75,
                    "recall": 0.74,
                    "f1": 0.73,
                    "AUROC": 0.78,
                },
                {
                    "id": "Trial 1B",
                    "status": "completed",
                    "test_fold_id": 1,
                    "accuracy": 0.74,
                    "precision": 0.73,
                    "recall": 0.72,
                    "f1": 0.71,
                    "AUROC": 0.76,
                },
                {
                    "id": "Trial 2A",
                    "status": "completed",
                    "test_fold_id": 2,
                    "accuracy": 0.91,
                    "precision": 0.90,
                    "recall": 0.89,
                    "f1": 0.88,
                    "AUROC": 0.93,
                },
                {
                    "id": "Trial 3A",
                    "status": "completed",
                    "test_fold_id": 3,
                    "accuracy": 0.65,
                    "precision": 0.64,
                    "recall": 0.63,
                    "f1": 0.62,
                    "AUROC": 0.67,
                },
                {
                    "id": "Trial 4A",
                    "status": "completed",
                    "test_fold_id": 4,
                    "accuracy": 0.88,
                    "precision": 0.87,
                    "recall": 0.86,
                    "f1": 0.85,
                    "AUROC": 0.89,
                },
                {
                    "id": "Trial skipped",
                    "status": "skipped",
                    "test_fold_id": 4,
                    "accuracy": 0.99,
                    "precision": 0.99,
                    "recall": 0.99,
                    "f1": 0.99,
                    "AUROC": 0.99,
                },
            ]
        )
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "feature_a",
                    "Description": "Feature A desc",
                    "Pandas Command": "df['a']",
                    "Status": "Active",
                },
                {
                    "Attribute": "feature_b",
                    "Description": "Feature B desc",
                    "Pandas Command": "df['b']",
                    "Status": "Pruned",
                },
                {
                    "Attribute": "feature_c",
                    "Description": "Feature C desc",
                    "Pandas Command": "df['c']",
                    "Status": "Active",
                },
            ]
        )

        aggregation = network.compute_fold_aggregation()

        assert aggregation["fold_count"] == 5
        assert [row["id"] for row in aggregation["best_iterations"]] == [
            "Trial 0B",
            "Trial 1A",
            "Trial 2A",
            "Trial 3A",
            "Trial 4A",
        ]
        assert aggregation["accuracy"]["mean"] == pytest.approx(
            (0.82 + 0.76 + 0.91 + 0.65 + 0.88) / 5
        )
        assert aggregation["accuracy"]["std"] == pytest.approx(
            pd.Series([0.82, 0.76, 0.91, 0.65, 0.88]).std(ddof=0)
        )
        assert [feature["name"] for feature in aggregation["active_features"]] == [
            "feature_a",
            "feature_c",
        ]

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_compute_fold_aggregation_regression_uses_lowest_nrmse_per_fold():
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.cfg = SimpleNamespace(
            modality="tabular",
            task_type="regression",
            experiment_name="demo-regression",
        )
        network.df_test_pool = pd.DataFrame(
            [
                {
                    "id": "Fold0-A",
                    "status": "completed",
                    "test_fold_id": 0,
                    "n_RMSE": 0.31,
                    "RMSE": 31.0,
                    "MSE": 961.0,
                    "MAE": 22.0,
                    "R2": 0.72,
                },
                {
                    "id": "Fold0-B",
                    "status": "completed",
                    "test_fold_id": 0,
                    "n_RMSE": 0.24,
                    "RMSE": 24.0,
                    "MSE": 576.0,
                    "MAE": 18.0,
                    "R2": 0.79,
                },
                {
                    "id": "Fold1-A",
                    "status": "completed",
                    "test_fold_id": 1,
                    "n_RMSE": 0.19,
                    "RMSE": 19.0,
                    "MSE": 361.0,
                    "MAE": 14.0,
                    "R2": 0.85,
                },
                {
                    "id": "Fold2-A",
                    "status": "completed",
                    "test_fold_id": 2,
                    "n_RMSE": 0.27,
                    "RMSE": 27.0,
                    "MSE": 729.0,
                    "MAE": 20.0,
                    "R2": 0.77,
                },
                {
                    "id": "Fold3-A",
                    "status": "completed",
                    "test_fold_id": 3,
                    "n_RMSE": 0.22,
                    "RMSE": 22.0,
                    "MSE": 484.0,
                    "MAE": 16.0,
                    "R2": 0.82,
                },
                {
                    "id": "Fold4-A",
                    "status": "completed",
                    "test_fold_id": 4,
                    "n_RMSE": 0.29,
                    "RMSE": 29.0,
                    "MSE": 841.0,
                    "MAE": 21.0,
                    "R2": 0.75,
                },
                {
                    "id": "Fold4-B",
                    "status": "completed",
                    "test_fold_id": 4,
                    "n_RMSE": 0.21,
                    "RMSE": 21.0,
                    "MSE": 441.0,
                    "MAE": 15.0,
                    "R2": 0.83,
                },
            ]
        )
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "signal_x",
                    "Description": "Signal X",
                    "Pandas Command": "df['x']",
                    "Status": "Active",
                },
            ]
        )

        aggregation = network.compute_fold_aggregation()

        assert aggregation["fold_count"] == 5
        assert [row["id"] for row in aggregation["best_iterations"]] == [
            "Fold0-B",
            "Fold1-A",
            "Fold2-A",
            "Fold3-A",
            "Fold4-B",
        ]
        assert aggregation["n_RMSE"]["mean"] == pytest.approx(
            (0.24 + 0.19 + 0.27 + 0.22 + 0.21) / 5
        )
        assert aggregation["n_RMSE"]["std"] == pytest.approx(
            pd.Series([0.24, 0.19, 0.27, 0.22, 0.21]).std(ddof=0)
        )
        assert aggregation["active_features"][0]["name"] == "signal_x"

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_compute_fold_aggregation_returns_empty_when_any_fold_is_missing():
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.cfg = SimpleNamespace(
            modality="tabular",
            task_type="classification",
            experiment_name="missing-fold-demo",
        )
        network.df_test_pool = pd.DataFrame(
            [
                {
                    "id": "Trial 0",
                    "status": "completed",
                    "test_fold_id": 0,
                    "accuracy": 0.84,
                    "precision": 0.80,
                    "recall": 0.82,
                    "f1": 0.81,
                    "AUROC": 0.91,
                },
                {
                    "id": "Trial 2",
                    "status": "completed",
                    "test_fold_id": 2,
                    "accuracy": 0.79,
                    "precision": 0.77,
                    "recall": 0.78,
                    "f1": 0.77,
                    "AUROC": 0.88,
                },
                {
                    "id": "Trial 4",
                    "status": "completed",
                    "test_fold_id": 4,
                    "accuracy": 0.81,
                    "precision": 0.80,
                    "recall": 0.79,
                    "f1": 0.79,
                    "AUROC": 0.87,
                },
            ]
        )
        network.df_attributes_explanations = pd.DataFrame(
            [
                {
                    "Attribute": "feature_a",
                    "Description": "Feature A desc",
                    "Pandas Command": "df['a']",
                    "Status": "Active",
                },
            ]
        )

        aggregation = network.compute_fold_aggregation()

        assert aggregation == {}

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_main_aborts_tabular_dataset_after_first_aborted_fold(tmp_path: Path):
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.cfg = SimpleNamespace(
            modality="tabular",
            task_type="classification",
            experiment_name="abort-demo",
            output_dir=tmp_path,
        )
        network.enable_wandb = False
        network.df_test_pool = pd.DataFrame()
        network.df_scientist_feedback_pool = pd.DataFrame()
        network.df_entities_attributes = pd.DataFrame({"id": []})
        network.df_attributes_explanations = pd.DataFrame(
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
        network._tabular_skip_reason = None
        network._dataset_run_status = "running"
        network._dataset_failure_reason = None

        async def fake_forward_pass(index: int) -> str:
            statuses = ["completed", "completed", "aborted", "completed", "completed"]
            status = statuses[index]
            row = {
                "id": f"Trial {index}",
                "status": status,
                "test_fold_id": index,
                "aborted_fold_id": index if status == "aborted" else None,
            }
            network.df_test_pool = pd.concat(
                [network.df_test_pool, pd.DataFrame([row])],
                ignore_index=True,
            )
            return status

        async def fake_save_data() -> None:
            return None

        async def fake_report_fold_aggregation_to_wandb(_aggregation: dict) -> None:
            return None

        async def fake_log_fold_aggregation_artifacts(_csv_paths: list[Path]) -> None:
            return None

        network.forward_pass = fake_forward_pass
        network.save_data = fake_save_data
        network.compute_fold_aggregation = lambda: pytest.fail(
            "aggregation must not run after an aborted fold"
        )
        network.print_fold_aggregation_to_terminal = lambda aggregation: pytest.fail(
            "aggregation output must not print after an aborted fold"
        )
        network.save_fold_aggregation_csv = lambda aggregation: pytest.fail(
            "aggregation artifacts must not be saved after an aborted fold"
        )
        network.report_fold_aggregation_to_wandb = fake_report_fold_aggregation_to_wandb
        network.log_fold_aggregation_artifacts = fake_log_fold_aggregation_artifacts

        with pytest.raises(RuntimeError, match="aborted"):
            asyncio.run(network.main())

        assert network.df_test_pool["status"].tolist() == [
            "completed",
            "completed",
            "aborted",
        ]

    except ImportError:
        pytest.skip("rogue_one_main not available")


@pytest.mark.integration
def test_save_fold_aggregation_csv_writes_summary_best_and_feature_files(
    tmp_path: Path,
):
    try:
        from rogue_one_main import RogueOneAgentNetwork

        network = RogueOneAgentNetwork.__new__(RogueOneAgentNetwork)
        network.cfg = SimpleNamespace(
            output_dir=tmp_path,
        )

        aggregation = {
            "dataset_id": "demo-dataset",
            "task_type": "classification",
            "fold_count": 5,
            "feature_count": 2,
            "best_iterations": [
                {"test_fold_id": 0, "id": "Trial 0", "accuracy": 0.81},
                {"test_fold_id": 1, "id": "Trial 1", "accuracy": 0.84},
            ],
            "active_features": [
                {
                    "name": "feature_a",
                    "description": "Feature A",
                    "pandas_command": "df['a']",
                },
                {
                    "name": "feature_b",
                    "description": "Feature B",
                    "pandas_command": "df['b']",
                },
            ],
            "accuracy": {"mean": 0.825, "std": 0.015},
            "precision": {"mean": 0.79, "std": 0.01},
            "recall": {"mean": 0.77, "std": 0.02},
            "f1": {"mean": 0.78, "std": 0.018},
            "AUROC": {"mean": 0.86, "std": 0.012},
        }

        csv_paths = network.save_fold_aggregation_csv(aggregation)

        assert len(csv_paths) == 3
        assert all(path.exists() for path in csv_paths)

        summary_df = pd.read_csv(tmp_path / "fold_aggregation_summary.csv")
        best_df = pd.read_csv(tmp_path / "fold_aggregation_best_iterations.csv")
        features_df = pd.read_csv(tmp_path / "fold_aggregation_active_features.csv")

        assert set(summary_df["metric"].tolist()) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "AUROC",
        }
        assert summary_df.loc[summary_df["metric"] == "accuracy", "mean"].iloc[
            0
        ] == pytest.approx(0.825)
        assert best_df["id"].tolist() == ["Trial 0", "Trial 1"]
        assert features_df["name"].tolist() == ["feature_a", "feature_b"]

    except ImportError:
        pytest.skip("rogue_one_main not available")
